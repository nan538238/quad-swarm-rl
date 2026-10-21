"""Four-UAV takeover patrol, infrastructure protocol E0001 (not paper evidence).

This is a custom multi-agent API with Gymnasium-style reset/step returns and a
scalar cooperative reward. Actions have shape (4, 3), clipped to [-1, 1], and
add bounded world-frame velocity corrections to each UAV's own slot tracker.
The inherited QuadSwarm nonlinear position controller converts the desired
velocity to motor thrust; QuadrotorDynamics integrates the real 6-DoF state.
No kinematic position updates or learned/privileged low-level policies are used.

The diagnostic world is obstacle-free, full-neighbor observation, zero motor
noise, and fixed dynamics. Contact is terminal; this task does not claim a
physical inter-UAV impact model. Separation is checked at every physics step.

Actor observation per agent (41 float32 values), in this exact order:
  velocity[3], own nominal slot minus position[3], fixed formation slot[3],
  nominal patrol velocity[3], then each other UAV in ascending fixed-ID order:
  relative position[3], relative velocity[3], current takeover flag[1]; then
  own takeover flag[1], elapsed current takeover seconds[1], phase one-hot[3]
  (patrol, takeover, handback), last executed desired velocity[3].
No operator target, duration, next action, or future release time is included.
Info is a diagnostic channel, never an actor input. Recovery rollouts are
explicit opt-in computations and their simulated steps are returned for cost
accounting. They are not automatically run by reset or step.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces

from gym_art.quadrotor_multi.quad_models import crazyflie_params
from gym_art.quadrotor_multi.quadrotor_control import (
    NonlinearPositionController,
    quadrotor_jacobian,
)
from gym_art.quadrotor_multi.quadrotor_dynamics import QuadrotorDynamics


@dataclass(frozen=True)
class TakeoverConfig:
    physics_dt: float = 0.02
    control_dt: float = 0.1
    episode_seconds: float = 24.0
    patrol_speed: float = 0.30
    altitude: float = 2.5
    slot_spacing: float = 1.3
    residual_velocity: float = 0.6
    max_velocity: float = 1.2
    slot_gain: float = 0.8
    takeover_enabled: bool = True
    takeover_start_range: tuple = (5.0, 6.0)
    takeover_duration_range: tuple = (4.0, 5.0)
    detour_distance_range: tuple = (1.0, 1.4)
    operator_speed: float = 0.7
    check_radius: float = 0.25
    check_hold_seconds: float = 0.4
    collision_distance: float = 0.25
    recovery_position_tolerance: float = 0.30
    recovery_velocity_tolerance: float = 0.20
    recovery_hold_seconds: float = 0.5
    recovery_timeout: float = 10.0

    def __post_init__(self):
        positive = (self.physics_dt, self.control_dt, self.episode_seconds,
                    self.patrol_speed, self.slot_spacing, self.residual_velocity,
                    self.max_velocity, self.slot_gain, self.operator_speed,
                    self.recovery_hold_seconds, self.recovery_timeout)
        if min(positive) <= 0:
            raise ValueError("Time steps, gains, speeds and scale parameters must be positive")
        ratio = self.control_dt / self.physics_dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if self.collision_distance >= self.slot_spacing:
            raise ValueError("Nominal slots must be outside the collision threshold")
        for bounds in (self.takeover_start_range, self.takeover_duration_range,
                       self.detour_distance_range):
            if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError("Invalid scenario range")
        if self.takeover_duration_range[0] < self.control_dt:
            raise ValueError("Takeover must last at least one control interval")


class _ZeroMotorNoise:
    """Explicit noise-free diagnostic; avoids touching NumPy's global RNG."""

    def noise(self):
        return np.zeros(4)


class _NumpyPositionController(NonlinearPositionController):
    """Use upstream NumPy step unchanged, without its unconditional TF import.

    These initial values are identical to the upstream tf_control=False path.
    No controller equations or gains are changed.
    """

    def __init__(self, dynamics):
        self.Jinv = np.linalg.inv(quadrotor_jacobian(dynamics))
        self.action = None
        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0
        self.rot_des = np.eye(3)
        self.tf_control = False


def _limit_rows(value, bound):
    return value * np.minimum(1.0, bound / np.maximum(
        np.linalg.norm(value, axis=-1, keepdims=True), 1e-12))


class TakeoverPatrolEnv:
    num_agents = 4
    obs_dim = 41
    action_dim = 3

    def __init__(self, config: Optional[TakeoverConfig] = None):
        self.config = config or TakeoverConfig()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4, 3), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4, self.obs_dim),
                                            dtype=np.float32)
        self.rng = np.random.default_rng()
        self.dynamics = []
        self.controllers = []
        self._initialized = False

    @property
    def time(self):
        return self.physics_steps * self.config.physics_dt

    @property
    def positions(self):
        return np.array([d.pos for d in self.dynamics])

    @property
    def velocities(self):
        return np.array([d.vel for d in self.dynamics])

    @property
    def nominal_velocity(self):
        return np.array([self.config.patrol_speed, 0.0, 0.0])

    @property
    def slot_targets(self):
        return self.slots + np.array([self.config.patrol_speed * self.time,
                                      0.0, self.config.altitude])

    @property
    def takeover_flags(self):
        flags = np.zeros(4)
        if self.phase == "takeover":
            flags[self.takeover_agent] = 1.0
        return flags

    def _sample_boundary_time(self, limits):
        low = int(np.ceil(limits[0] / self.config.control_dt - 1e-9))
        high = int(np.floor(limits[1] / self.config.control_dt + 1e-9))
        if high < low:
            raise ValueError("Scenario range contains no control boundary")
        return int(self.rng.integers(low, high + 1)) * self.config.control_dt

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Use a TakeoverConfig for explicit scenario changes")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scenario_seed = seed
        c = self.config
        h = c.slot_spacing / 2
        self.slots = np.array([[-h, -h, 0], [-h, h, 0],
                               [h, -h, 0], [h, h, 0]], dtype=float)
        self.physics_steps = 0
        self.control_steps = 0
        self.phase = "patrol"
        self.takeover_agent = int(self.rng.integers(4))
        self._takeover_start = self._sample_boundary_time(c.takeover_start_range)
        self._takeover_end = self._takeover_start + self._sample_boundary_time(
            c.takeover_duration_range)
        self._detour = float(self.rng.uniform(*c.detour_distance_range))
        self._operator_target = None
        self._wait_positions = None
        self.handover_time = None
        self.recovery_time = None
        self.recovery_success = False
        self._recovery_hold = 0.0
        self._recovery_decided = False
        self._check_hold = 0.0
        self.check_completed = False
        self.collision = False
        self.min_pair_distance = float("inf")
        self.terminated = False
        self.truncated = False
        self.last_requested_velocity = np.zeros((4, 3))
        self.last_executed_velocity = np.zeros((4, 3))
        self.last_override_mask = np.zeros(4, dtype=bool)
        self.dynamics, self.controllers = [], []
        params = crazyflie_params()
        params["noise"]["thrust_noise_ratio"] = 0.0
        for pos in self.slot_targets:
            dyn = QuadrotorDynamics(deepcopy(params), room_box=[[-100, -30, 0],
                                    [100, 30, 15]], dt=c.physics_dt, use_numba=False)
            dyn.thrust_noise = _ZeroMotorNoise()
            dyn.set_state(pos, self.nominal_velocity, np.eye(3), np.zeros(3))
            # Start with hover rotor filter state, not an artificial zero-thrust drop.
            dyn.thrust_cmds_damp[:] = 1.0 / dyn.thrust_to_weight
            dyn.thrust_rot_damp[:] = np.sqrt(dyn.thrust_cmds_damp)
            self.dynamics.append(dyn)
            self.controllers.append(_NumpyPositionController(dyn))
        self._initial_mean_x = float(self.positions[:, 0].mean())
        self._initialized = True
        self._update_phase()
        return self.observe(), self._info()

    def _update_phase(self):
        if not self.config.takeover_enabled:
            return
        if self.phase == "patrol" and self.time + 1e-9 >= self._takeover_start:
            self.phase = "takeover"
            self._wait_positions = self.positions.copy()
            self._operator_target = self.positions[self.takeover_agent].copy()
            # Detour outward from this UAV's assigned side; same policy across methods.
            side = np.sign(self.slots[self.takeover_agent, 1])
            self._operator_target += np.array([0.2, side * self._detour, 0.2])
        if self.phase == "takeover" and self.time + 1e-9 >= self._takeover_end:
            self.force_handback()

    def force_handback(self):
        """External handback event; labels use this only on restored copies."""
        self.phase = "handback"
        self.handover_time = self.time
        self.recovery_time = None
        self.recovery_success = False
        self._recovery_hold = 0.0
        self._recovery_decided = False

    def observe(self):
        if not self._initialized:
            raise RuntimeError("reset must be called before observe")
        pos, vel, flags = self.positions, self.velocities, self.takeover_flags
        elapsed = max(0.0, self.time - self._takeover_start) if self.phase == "takeover" else 0.0
        phase = np.eye(3)[("patrol", "takeover", "handback").index(self.phase)]
        rows = []
        for i in range(4):
            values = [vel[i], self.slot_targets[i] - pos[i], self.slots[i],
                      self.nominal_velocity]
            for j in range(4):
                if i != j:
                    values.extend((pos[j] - pos[i], vel[j] - vel[i], [flags[j]]))
            values.extend(([flags[i]], [elapsed], phase, self.last_executed_velocity[i]))
            rows.append(np.concatenate(values))
        return np.asarray(rows, dtype=np.float32)

    def nominal_commands(self):
        return _limit_rows(self.nominal_velocity + self.config.slot_gain * (
            self.slot_targets - self.positions), self.config.max_velocity)

    def reference_actions(self, mode="nominal"):
        """Simple decentralized diagnostics; wait uses each UAV's own frozen pose."""
        if mode == "nominal":
            return np.zeros((4, 3), dtype=np.float32)
        if mode != "wait":
            raise ValueError("reference mode must be nominal or wait")
        if self.phase != "takeover":
            return np.zeros((4, 3), dtype=np.float32)
        desired = self.config.slot_gain * (self._wait_positions - self.positions)
        return np.clip((desired - self.nominal_commands()) / self.config.residual_velocity,
                       -1, 1).astype(np.float32)

    def operator_command(self):
        """Overridable external velocity interface; default arithmetic unchanged."""
        i = self.takeover_agent
        return _limit_rows((self._operator_target - self.positions[i])[None, :],
                           self.config.operator_speed)[0]

    def step(self, actions):
        if not self._initialized or self.terminated or self.truncated:
            raise RuntimeError("reset required before step or after episode completion")
        action = np.asarray(actions, dtype=float)
        if action.shape != (4, 3) or not np.isfinite(action).all():
            raise ValueError("actions must be finite with shape (4, 3)")
        action = np.clip(action, -1, 1)
        c = self.config
        before_x = float(self.positions[:, 0].mean())
        self.last_override_mask = np.zeros(4, dtype=bool)
        for _ in range(round(c.control_dt / c.physics_dt)):
            self._update_phase()
            requested = _limit_rows(self.nominal_commands() + c.residual_velocity * action,
                                    c.max_velocity)
            executed = requested.copy()
            if self.phase == "takeover":
                i = self.takeover_agent
                executed[i] = self.operator_command()
                self.last_override_mask[i] = True
            self.last_requested_velocity = requested.copy()
            self.last_executed_velocity = executed.copy()
            for i, (dyn, ctrl) in enumerate(zip(self.dynamics, self.controllers)):
                # Algebraic velocity target through the unchanged position controller:
                # kp*(goal-pos)-kd*v == kd*(desired_velocity-v).
                goal = dyn.pos + (ctrl.kd_p / ctrl.kp_p) * executed[i]
                ctrl.step(dyn, goal, c.physics_dt)
            self.physics_steps += 1
            self._update_safety_and_metrics()
            if self.collision:
                self.terminated = True
                break
            if self.time + 1e-9 >= c.episode_seconds:
                self.truncated = True
                break
        self.control_steps += 1
        self._update_phase()
        progress = float(self.positions[:, 0].mean()) - before_x
        # Stationary waiting earns no progress reward and retains tracking cost.
        tracking = np.linalg.norm(self.slot_targets - self.positions, axis=1)
        active = self.takeover_flags == 0
        reward = progress / c.patrol_speed - 0.05 * c.control_dt * float(tracking[active].mean())
        reward -= 20.0 * float(self.collision)
        return self.observe(), float(reward), self.terminated, self.truncated, self._info()

    def _update_safety_and_metrics(self):
        pos = self.positions
        pair = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        distance = float(pair[np.triu_indices(4, 1)].min())
        self.min_pair_distance = min(self.min_pair_distance, distance)
        contact = distance < self.config.collision_distance
        room_contact = any(d.crashed_floor or d.crashed_wall or d.crashed_ceiling
                           for d in self.dynamics)
        invalid = not (np.isfinite(pos).all() and np.isfinite(self.velocities).all())
        self.collision = self.collision or bool(contact or room_contact or invalid)
        if self.phase == "takeover":
            close = np.linalg.norm(pos[self.takeover_agent] - self._operator_target) <= self.config.check_radius
            self._check_hold = self._check_hold + self.config.physics_dt if close else 0.0
            self.check_completed |= self._check_hold + 1e-9 >= self.config.check_hold_seconds
        self._update_recovery()

    def _update_recovery(self):
        if self.phase != "handback" or self._recovery_decided:
            return
        c = self.config
        elapsed = self.time - self.handover_time
        pos_ok = np.max(np.linalg.norm(self.slot_targets - self.positions, axis=1)) <= c.recovery_position_tolerance
        vel_ok = np.max(np.linalg.norm(self.velocities - self.nominal_velocity, axis=1)) <= c.recovery_velocity_tolerance
        self._recovery_hold = self._recovery_hold + c.physics_dt if pos_ok and vel_ok and not self.collision else 0.0
        if self._recovery_hold + 1e-9 >= c.recovery_hold_seconds and elapsed <= c.recovery_timeout + 1e-9:
            self.recovery_success = True
            self.recovery_time = elapsed
            self._recovery_decided = True
        elif self.collision or elapsed + 1e-9 >= c.recovery_timeout:
            self._recovery_decided = True

    def metrics(self):
        elapsed = None if self.handover_time is None else self.time - self.handover_time
        censored = self.handover_time is not None and not self.recovery_success
        return {
            "time": self.time,
            "patrol_progress": float(self.positions[:, 0].mean()) - self._initial_mean_x,
            "nominal_patrol_progress": self.config.patrol_speed * self.time,
            "patrol_progress_loss": self.config.patrol_speed * self.time - (
                float(self.positions[:, 0].mean()) - self._initial_mean_x),
            "collision": bool(self.collision),
            "min_pair_distance": None if np.isinf(self.min_pair_distance) else self.min_pair_distance,
            "check_completed": bool(self.check_completed),
            "handover_observed": self.handover_time is not None,
            "recovery_success": bool(self.recovery_success),
            "recovery_time": self.recovery_time,
            "recovery_time_with_timeout": (self.recovery_time if self.recovery_success else
                                           self.config.recovery_timeout) if elapsed is not None else None,
            "recovery_censored": censored,
            "recovery_observation_seconds": elapsed,
            "recovery_decided": bool(self._recovery_decided or (
                censored and (self.terminated or self.truncated))),
            "physics_steps": self.physics_steps,
            "control_steps": self.control_steps,
        }

    def _info(self):
        return {**self.metrics(), "phase": self.phase,
                "positions": self.positions.copy(), "velocities": self.velocities.copy(),
                "takeover_flags": self.takeover_flags.copy(),
                "override_mask": self.last_override_mask.copy(),
                "requested_velocity": self.last_requested_velocity.copy(),
                "executed_velocity": self.last_executed_velocity.copy()}

    def snapshot(self):
        """Full state including RNG, motor filters, rotation/SVD clock and metrics.

        Snapshot dynamics dictionaries explicitly: upstream __deepcopy__ rebuilds
        its noise object and would otherwise silently lose noise state.
        """
        plain = {k: v for k, v in self.__dict__.items() if k != "dynamics"}
        return deepcopy({"env": plain, "dynamics": [d.__dict__ for d in self.dynamics]})

    def restore(self, snapshot):
        state = deepcopy(snapshot)
        self.__dict__.clear()
        self.__dict__.update(state["env"])
        self.dynamics = []
        for attrs in state["dynamics"]:
            dyn = QuadrotorDynamics.__new__(QuadrotorDynamics)
            dyn.__dict__.update(attrs)
            self.dynamics.append(dyn)

    def evaluate_immediate_handback(self, max_seconds=None):
        """Opt-in fixed zero-residual reference rollout; never modifies live state.

        The returned time is failure-censored at the requested horizon. If a
        shorter-than-protocol horizon is requested, ``protocol_horizon_complete``
        explicitly marks this; it cannot be treated as a protocol timeout.
        """
        if not self._initialized or self.terminated or self.truncated:
            raise RuntimeError("Labels require a live initialized state")
        horizon = self.config.recovery_timeout if max_seconds is None else float(max_seconds)
        if horizon <= 0:
            raise ValueError("max_seconds must be positive")
        horizon = min(horizon, self.config.recovery_timeout)
        saved = self.snapshot()
        start_steps = self.physics_steps
        try:
            self.force_handback()
            start = self.time
            # Counterfactual recovery uses its own horizon, independent of the
            # source episode's remaining time; all other configuration is fixed.
            from dataclasses import replace
            self.config = replace(self.config, episode_seconds=start + horizon)
            while self.time - start < horizon - 1e-9 and not self._recovery_decided:
                _, _, done, truncated, _ = self.step(self.reference_actions("nominal"))
                if done or truncated:
                    break
            return {"recovery_success": bool(self.recovery_success),
                    "recovery_time": self.recovery_time,
                    "time_with_timeout": self.recovery_time if self.recovery_success else horizon,
                    "collision": bool(self.collision),
                    "simulated_physics_steps": self.physics_steps - start_steps,
                    "rollout_horizon": horizon,
                    "protocol_horizon_complete": horizon >= saved["env"]["config"].recovery_timeout,
                    "reference_controller": "nominal_zero_residual_v1"}
        finally:
            self.restore(saved)

    def close(self):
        pass
