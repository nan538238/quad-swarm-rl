"""Causal takeover adaptation pilot, separate from the frozen CP2 experiment.

All actors act throughout an episode except the externally overridden UAV.
History is row-local and reset at every episode; no future operator information
is exposed. The three variants share architecture, PPO and reward settings.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .env import _limit_rows
from .geometry import GeometryConfig, GeometryTakeoverEnv
from .policy import observation_scale, squashed_log_prob


VARIANTS = ('blind', 'flag', 'history')
AUTHORITY_COLUMNS = [18, 25, 32, 33, 34, 35, 36, 37]
PROFILES = {
    'steady': (0.70, 0, 0.0),
    'delayed': (0.55, 2, 0.05),
    'noisy': (0.85, 1, 0.12),
    'unseen_slow': (0.40, 3, 0.08),
    'unseen_fast': (1.00, 0, 0.16),
}


@dataclass(frozen=True)
class AdaptationConfig(GeometryConfig):
    command_delay_steps: int = 0
    command_noise_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.command_delay_steps < 0 or not np.isfinite(self.command_noise_std) or self.command_noise_std < 0:
            raise ValueError('Invalid simulated-operator profile')


class AdaptationEnv(GeometryTakeoverEnv):
    def reset(self, *, seed=None, options=None):
        self.phase_min_distance = {key: None for key in ('patrol', 'takeover', 'handback')}
        self.operator_rng = np.random.default_rng(np.random.SeedSequence([seed or 0, 9187]))
        self._command_queue = deque()
        self._command_step = -1
        self._held_command = np.zeros(3)
        return super().reset(seed=seed, options=options)

    def _update_safety_and_metrics(self):
        super()._update_safety_and_metrics()
        positions = self.positions
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        current = float(distances[np.triu_indices(4, 1)].min())
        previous = self.phase_min_distance[self.phase]
        self.phase_min_distance[self.phase] = current if previous is None else min(current, previous)

    def operator_command(self):
        # Sample once per control interval, never once per physics substep.
        if self._command_step != self.control_steps:
            command = super().operator_command()
            command += self.operator_rng.normal(0, self.config.command_noise_std, 3)
            command = _limit_rows(command[None], self.config.operator_speed)[0]
            self._command_queue.append(command)
            self._held_command = (self._command_queue.popleft()
                                  if len(self._command_queue) > self.config.command_delay_steps
                                  else np.zeros(3))
            self._command_step = self.control_steps
        return self._held_command.copy()


def make_env(geometry, profile, *, manual=False):
    speed, delay, noise = PROFILES[profile]
    cls = ManualTakeoverEnv if manual else AdaptationEnv
    if geometry not in ('outward', 'crossing'):
        raise ValueError(geometry)
    return cls(AdaptationConfig(detour_angle_degrees=0 if geometry == 'outward' else 180,
        detour_distance_range=(1.8, 2.4), operator_speed=speed,
        command_delay_steps=delay, command_noise_std=noise))


class ManualTakeoverEnv(AdaptationEnv):
    """Real operator controls start/end and world-frame velocity; no auto timer."""
    def reset(self, *, seed=None, options=None):
        self.manual_velocity = np.zeros(3)
        return super().reset(seed=seed, options=options)

    def _update_phase(self):
        # Human events alone change authority in manual mode.
        return

    def begin_takeover(self):
        if self.phase != 'patrol' or self.terminated or self.truncated:
            raise ValueError('Takeover requires a live patrol episode')
        self._takeover_start = self.time
        self.phase = 'takeover'
        self._wait_positions = self.positions.copy()
        side = np.sign(self.slots[self.takeover_agent, 1])
        angle = np.deg2rad(self.config.detour_angle_degrees)
        self._operator_target = self.positions[self.takeover_agent] + np.array(
            [0.2 + np.sin(angle) * self._detour, side * np.cos(angle) * self._detour, 0.2])

    def force_handback(self):
        if self.phase != 'takeover':
            raise ValueError('Handback requires an active takeover')
        self.manual_velocity[:] = 0
        super().force_handback()

    def operator_command(self):
        return _limit_rows(np.asarray(self.manual_velocity, dtype=float)[None],
                           self.config.max_velocity)[0]


class ObservationHistory:
    def __init__(self, variant, frames=4):
        if variant not in VARIANTS or frames < 1:
            raise ValueError('Invalid observation variant/history length')
        self.variant, self.frames = variant, frames
        self.buffer = deque(maxlen=frames)

    def reset(self, observations):
        self.buffer.clear()
        return self.push(observations)

    def push(self, observations):
        obs = np.asarray(observations, dtype=np.float32).copy()
        if obs.shape != (4, 41) or not np.isfinite(obs).all():
            raise ValueError('Expected finite row-local observations')
        if self.variant == 'blind':
            obs[:, AUTHORITY_COLUMNS] = 0
        self.buffer.appendleft(obs)
        history = list(self.buffer) if self.variant == 'history' else [obs]
        return np.concatenate(history + [np.zeros_like(obs)] * (self.frames - len(history)), axis=-1)


class AdaptationPolicy(nn.Module):
    def __init__(self, frames=4):
        super().__init__()
        self.frames = frames
        dimension = 41 * frames
        self.register_buffer('observation_scale', torch.from_numpy(np.tile(observation_scale(), frames)))
        self.actor = nn.Sequential(nn.Linear(dimension, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        self.critic = nn.Sequential(nn.Linear(4 * dimension, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def normalize(self, obs):
        if obs.shape[-1] != 41 * self.frames:
            raise ValueError('Wrong observation history dimension')
        return (obs / self.observation_scale).clamp(-10, 10)

    def distribution(self, obs):
        mean = self.actor(self.normalize(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        if obs.shape[-2:] != (4, 41 * self.frames):
            raise ValueError('Central critic needs all four row-local histories')
        return self.critic(self.normalize(obs).flatten(-2)).squeeze(-1)

    def deterministic(self, obs, autonomous):
        return torch.where(autonomous.unsqueeze(-1), self.distribution(obs).mean.tanh(), 0.0)

    def sample(self, obs, autonomous, generator):
        distribution = self.distribution(obs)
        latent = distribution.mean + distribution.stddev * torch.randn(
            distribution.mean.shape, generator=generator)
        actions = torch.where(autonomous.unsqueeze(-1), latent.tanh(), 0.0)
        return actions, latent, squashed_log_prob(distribution, latent), autonomous, self.value(obs)


def load_policy(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = AdaptationPolicy(saved['frames'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved
