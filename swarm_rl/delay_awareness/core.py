"""Four-UAV route task, causal timestamped channel, and fixed rule baselines."""
from collections import deque
from dataclasses import dataclass

import numpy as np

from swarm_rl.takeover.env import TakeoverConfig, TakeoverPatrolEnv, _limit_rows


@dataclass(frozen=True)
class RouteConfig(TakeoverConfig):
    takeover_enabled: bool = False
    episode_seconds: float = 20.0
    residual_velocity: float = 1.2
    arrival_radius: float = 0.3
    arrival_speed: float = 0.2
    arrival_hold: float = 0.4


class RouteEnv(TakeoverPatrolEnv):
    """New task/reward on the unchanged upstream 6-DoF dynamics/controller.

No policy receives the base task's privileged observation. Policies must use
DelayedChannel. Only own goal and own proprioception are available instantly.
"""
    def __init__(self, geometry='crossing', config=None):
        if geometry not in ('crossing', 'outward'):
            raise ValueError(geometry)
        self.geometry = geometry
        self.route_ready = False
        super().__init__(config or RouteConfig())

    @property
    def slot_targets(self):
        return self.goals if self.route_ready else super().slot_targets

    def observe(self):
        # Deliberately prohibit inherited instantaneous peer observations.
        return None

    def _info(self):
        return {'time': self.time, 'collision': self.collision}

    def reset(self, *, seed=None, options=None):
        self.route_ready = False
        super().reset(seed=seed, options=options)
        rng = np.random.default_rng(seed)
        angles = np.arange(4) * np.pi / 2 + rng.uniform(-np.pi, np.pi)
        radii = rng.uniform(2.8, 3.2, 4)
        start = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.full(4, 2.5)))
        self.goals = start.copy()
        direction = start[:, :2] / radii[:, None]
        self.goals[:, :2] += (-6 if self.geometry == 'crossing' else 6) * direction
        self.cruise_speeds = rng.uniform(0.7, 1.1, 4)
        self.launch_times = rng.uniform(0.0, 0.6, 4)
        for dyn, pos in zip(self.dynamics, start):
            dyn.set_state(pos, np.zeros(3), np.eye(3), np.zeros(3))
        self.route_ready = True
        self.path_length = 0.0
        self._hold = 0.0
        self.success = False
        self.completion_time = None
        self.collision_time = None
        self.last_executed_velocity[:] = 0
        self.last_requested_velocity[:] = 0
        return None, self.route_metrics()

    def nominal_commands(self):
        if not self.route_ready:
            return super().nominal_commands()
        commands = _limit_rows(self.config.slot_gain * (self.goals - self.positions),
                               self.cruise_speeds[:, None])
        commands[self.time + 1e-9 < self.launch_times] = 0
        return commands

    def _update_safety_and_metrics(self):
        super()._update_safety_and_metrics()
        if self.collision and self.collision_time is None:
            self.collision_time = self.time
        distances = np.linalg.norm(self.goals - self.positions, axis=1)
        arrived = ((distances <= self.config.arrival_radius) &
                   (np.linalg.norm(self.velocities, axis=1) <= self.config.arrival_speed)).all()
        self._hold = self._hold + self.config.physics_dt if arrived and not self.collision else 0.0
        if self._hold + 1e-9 >= self.config.arrival_hold and not self.success:
            self.success = True
            self.completion_time = self.time

    def step(self, actions):
        before = self.positions.copy()
        distance_before = np.linalg.norm(self.goals - before, axis=1).mean()
        _, _, terminated, truncated, _ = super().step(actions)
        self.path_length += float(np.linalg.norm(self.positions - before, axis=1).sum())
        self.terminated = terminated or self.success
        progress = distance_before - np.linalg.norm(self.goals - self.positions, axis=1).mean()
        reward = float(2 * progress - 0.02 - 20 * self.collision + 5 * (self.success and not self.collision))
        return None, reward, self.terminated, truncated, self.route_metrics()

    def route_metrics(self):
        success = bool(self.success and not self.collision)
        return {'success': success, 'collision': bool(self.collision), 'collision_time': self.collision_time,
            'completion_time': self.completion_time if success else None,
            'failure_inclusive_completion_cost': self.completion_time if success else self.config.episode_seconds,
            'observation_seconds': self.time, 'path_length_m': self.path_length,
            'goal_distance_mean': float(np.linalg.norm(self.goals - self.positions, axis=1).mean()),
            'min_distance': None if np.isinf(self.min_pair_distance) else float(self.min_pair_distance),
            'physics_steps': self.physics_steps, 'control_steps': self.control_steps}


class DelayedChannel:
    """Periodic broadcasts, fixed integer-step delay, synchronized timestamps.

All methods share an initial t=0 packet (preflight initialization). Until the
first delayed update, that packet ages normally. Sender velocity is measured;
there is no measurement noise or packet loss in this diagnostic.
"""
    dim = 36

    def __init__(self, delay_steps):
        if not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError('Delay must be a nonnegative integer number of control steps')
        self.delay_steps = delay_steps
        self.queue = deque()
        self.current_packet = None
        self.last_step = -1

    def capture(self, env):
        if env.control_steps < self.last_step:
            raise RuntimeError('Create a fresh channel at each episode reset')
        if env.control_steps != self.last_step:
            packet = (env.control_steps, env.time, env.positions.copy(), env.velocities.copy())
            self.queue.append(packet)
            if self.current_packet is None:
                self.current_packet = packet
            while self.queue and self.queue[0][0] <= env.control_steps - self.delay_steps:
                self.current_packet = self.queue.popleft()
            self.last_step = env.control_steps
        return self.current_packet

    def observation(self, env, estimate='raw'):
        if estimate not in ('raw', 'cv', 'oracle'):
            raise ValueError(estimate)
        _, sent_at, packet_pos, packet_vel = self.capture(env)
        age = env.time - sent_at
        own_pos, own_vel = env.positions, env.velocities
        if estimate == 'oracle':
            peer_pos, peer_vel, reported_age = own_pos, own_vel, 0.0
        else:
            peer_pos = packet_pos + (packet_vel * age if estimate == 'cv' else 0)
            peer_vel, reported_age = packet_vel, age
        rows, errors = [], []
        nominal = env.nominal_commands()
        for i in range(4):
            parts = [own_vel[i], env.goals[i] - own_pos[i], env.last_executed_velocity[i], nominal[i]]
            for j in range(4):
                if j == i:
                    continue
                parts.extend((peer_pos[j] - own_pos[i], peer_vel[j] - own_vel[i], [reported_age, 1.0]))
                errors.append(float(np.square(peer_pos[j] - own_pos[j]).sum()))
            rows.append(np.concatenate(parts))
        return np.asarray(rows, dtype=np.float32), {'packet_age': age,
            'peer_position_squared_error_sum': sum(errors), 'peer_estimates': len(errors)}


def peer_fields(obs):
    values = np.asarray(obs)
    if values.shape != (4, 36):
        raise ValueError('Expected four row-local 36-dimensional observations')
    neighbors = values[:, 12:].reshape(4, 3, 8)
    return neighbors[:, :, :3], neighbors[:, :, 3:6]


def reactive(obs, config):
    relative, _ = peer_fields(obs)
    distance = np.linalg.norm(relative, axis=-1, keepdims=True)
    away = np.divide(-relative, distance, out=np.zeros_like(relative), where=distance > 1e-9)
    correction = (1.5 * np.maximum(0, 0.9 - distance) * away).sum(1)
    return np.clip(correction / config.residual_velocity, -1, 1).astype(np.float32)


def predictive(obs, config):
    """Fixed candidate-velocity lookahead, not ORCA/MPC or a learned method.

Every candidate evaluates constant peer velocity over 1.2s with approximate
own acceleration response. Cost trades nominal tracking against short predicted
separation. Hyperparameters are fixed before the diagnostic and not fitted.
"""
    relative, relative_vel = peer_fields(obs)
    offsets = np.array([[x, y, 0.0] for x in (-1.0, -0.5, 0, 0.5, 1.0)
                        for y in (-1.0, -0.5, 0, 0.5, 1.0)], dtype=float)
    horizons = np.linspace(0.2, 1.2, 6)
    response_time = 0.25
    actions = []
    for i in range(4):
        own_velocity, desired = obs[i, :3], obs[i, 9:12]
        candidates = _limit_rows(desired[None] + offsets * config.residual_velocity, config.max_velocity)
        # Exponential acceleration surrogate; only own currently observed velocity.
        displacement = (candidates[:, None] * horizons[None, :, None] +
            (own_velocity - candidates)[:, None] * response_time *
            (1 - np.exp(-horizons / response_time))[None, :, None])
        neighbor_velocity = relative_vel[i] + own_velocity
        future = relative[i][None, :, None, :] + neighbor_velocity[None, :, None, :] * horizons[None, None, :, None]
        separation = np.linalg.norm(future - displacement[:, None], axis=-1)
        penalty = np.square(np.maximum(0, 0.65 - separation)).sum((1, 2))
        cost = np.square(candidates - desired).sum(1) + 40 * penalty
        choice = candidates[int(np.argmin(cost))]
        actions.append(np.clip((choice - desired) / config.residual_velocity, -1, 1))
    return np.asarray(actions, dtype=np.float32)


POLICIES = {
    'nominal': ('raw', None),
    'reactive_raw': ('raw', reactive),
    'reactive_cv': ('cv', reactive),
    'reactive_oracle': ('oracle', reactive),
    'predictive_raw': ('raw', predictive),
    'predictive_cv': ('cv', predictive),
    'predictive_oracle': ('oracle', predictive),
}
