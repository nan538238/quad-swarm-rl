"""E0002 fixed geometry sweep and observation-only reactive diagnostic policy.

These are task/control diagnostics, not a learned method. Geometry is hidden
from actors: only the scripted operator target changes at takeover onset.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from swarm_rl.takeover.env import TakeoverConfig, TakeoverPatrolEnv


@dataclass(frozen=True)
class GeometryConfig(TakeoverConfig):
    """Inherited defaults reproduce E0001; experiments set range explicitly."""

    detour_angle_degrees: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.detour_angle_degrees):
            raise ValueError("detour_angle_degrees must be finite")


class GeometryTakeoverEnv(TakeoverPatrolEnv):
    """Preserve dynamics/sampling and rotate only the hidden inspection offset."""

    def __init__(self, config: Optional[GeometryConfig] = None):
        super().__init__(config or GeometryConfig())
        self.collision_phase = None
        self.collision_time = None

    def reset(self, *, seed=None, options=None):
        self.collision_phase = None
        self.collision_time = None
        return super().reset(seed=seed, options=options)

    def _update_phase(self):
        before = self.phase
        super()._update_phase()
        if before != "patrol" or self.phase != "takeover":
            return
        angle_degrees = self.config.detour_angle_degrees
        # Retain the original target arithmetic exactly at the default angle.
        if angle_degrees % 360.0 == 0.0:
            return
        angle = np.deg2rad(angle_degrees)
        side = np.sign(self.slots[self.takeover_agent, 1])
        delta = np.array([0.2 + np.sin(angle) * self._detour,
                          side * np.cos(angle) * self._detour, 0.2])
        self._operator_target = self.positions[self.takeover_agent] + delta

    def _update_safety_and_metrics(self):
        super()._update_safety_and_metrics()
        if self.collision and self.collision_phase is None:
            # Capture the phase that generated this physics transition, before
            # step's final phase update can cross the handback boundary.
            self.collision_phase = self.phase
            self.collision_time = self.time


def reactive_actions(obs, gain=1.5, radius=0.9, residual_velocity=0.6):
    """Bounded repulsion from each actor's own current neighbor observations.

    Accepts one observation (41,) or any batch (..., 41). Each output row is
    independent of other rows. Neighbor positions are neighbor-minus-own, so
    their negated unit vectors point away from neighbors. Coincident positions
    contribute zero: no arbitrary privileged escape direction is introduced.
    Phase gating belongs to the evaluator, not this shared control function.
    """
    values = np.asarray(obs)
    if values.ndim < 1 or values.shape[-1] != TakeoverPatrolEnv.obs_dim:
        raise ValueError("obs must have final dimension 41")
    if (not np.isfinite(gain) or gain < 0 or not np.isfinite(radius)
            or radius <= 0 or not np.isfinite(residual_velocity)
            or residual_velocity <= 0):
        raise ValueError("gain must be nonnegative; radius and residual_velocity positive")
    relative = np.stack([values[..., start:start + 3]
                         for start in (12, 19, 26)], axis=-2).astype(float)
    if not np.isfinite(relative).all():
        raise ValueError("neighbor relative positions must be finite")
    distance = np.linalg.norm(relative, axis=-1, keepdims=True)
    unit_away = np.divide(-relative, distance, out=np.zeros_like(relative),
                          where=distance > 0)
    velocity = (gain * np.maximum(0.0, radius - distance) * unit_away).sum(axis=-2)
    return np.clip(velocity / residual_velocity, -1.0, 1.0).astype(np.float32)
