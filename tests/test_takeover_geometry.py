"""Semantic checks for the fixed E0002 diagnostic, without method claims."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from swarm_rl.takeover.env import TakeoverConfig, TakeoverPatrolEnv
from swarm_rl.takeover.geometry import GeometryConfig, GeometryTakeoverEnv, reactive_actions


class GeometryEnvironmentTests(unittest.TestCase):
    def test_defaults_preserve_original_trajectory_through_takeover_and_handback(self):
        old = TakeoverConfig(takeover_start_range=(0.2, 0.2),
                             takeover_duration_range=(0.4, 0.4))
        new = GeometryConfig(takeover_start_range=(0.2, 0.2),
                             takeover_duration_range=(0.4, 0.4))
        self.assertEqual(new.detour_distance_range, (1.0, 1.4))
        self.assertEqual(new.detour_angle_degrees, 0.0)
        with self.assertRaises(FrozenInstanceError):
            new.detour_angle_degrees = 90
        a, b = TakeoverPatrolEnv(old), GeometryTakeoverEnv(new)
        np.testing.assert_array_equal(a.reset(seed=101)[0], b.reset(seed=101)[0])
        for action in np.random.default_rng(4).uniform(-0.1, 0.1, (9, 4, 3)):
            result_a, result_b = a.step(action), b.step(action)
            np.testing.assert_array_equal(result_a[0], result_b[0])
            self.assertEqual(result_a[1:4], result_b[1:4])
            self.assertEqual(a.metrics(), b.metrics())
        self.assertEqual(a.phase, "handback")
        np.testing.assert_array_equal(a._operator_target, b._operator_target)

    def test_azimuth_hidden_at_takeover_and_preserves_other_sampling(self):
        config = dict(takeover_start_range=(0, 0), detour_distance_range=(1.0, 2.6))
        a = GeometryTakeoverEnv(GeometryConfig(**config, detour_angle_degrees=0))
        b = GeometryTakeoverEnv(GeometryConfig(**config, detour_angle_degrees=90))
        obs_a, _ = a.reset(seed=102)
        obs_b, _ = b.reset(seed=102)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(a.positions, b.positions)
        self.assertEqual(a.takeover_agent, b.takeover_agent)
        self.assertEqual(a._takeover_end, b._takeover_end)
        self.assertEqual(a._detour, b._detour)
        self.assertFalse(np.array_equal(a._operator_target, b._operator_target))
        b._operator_target += [1, 2, 3]
        np.testing.assert_array_equal(obs_b, b.observe())
        np.testing.assert_array_equal(a.rng.random(4), b.rng.random(4))

    def test_geometry_target_formula_and_seed_determinism(self):
        config = GeometryConfig(takeover_start_range=(0, 0),
                                detour_distance_range=(1.0, 2.6),
                                detour_angle_degrees=135)
        a, b = GeometryTakeoverEnv(config), GeometryTakeoverEnv(config)
        a.reset(seed=103)
        b.reset(seed=103)
        np.testing.assert_array_equal(a._operator_target, b._operator_target)
        angle = np.deg2rad(135)
        side = np.sign(a.slots[a.takeover_agent, 1])
        expected = a.positions[a.takeover_agent] + [
            0.2 + np.sin(angle) * a._detour,
            side * np.cos(angle) * a._detour, 0.2]
        np.testing.assert_array_equal(a._operator_target, expected)

    def test_collision_phase_captured_before_boundary_update_and_reset(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(0, 0),
                                                takeover_duration_range=(0.1, 0.1)))
        env.reset(seed=100)
        env.physics_steps = 5
        env.dynamics[0].pos = env.dynamics[1].pos.copy()
        env._update_safety_and_metrics()
        self.assertEqual(env.collision_phase, "takeover")
        self.assertAlmostEqual(env.collision_time, 0.1)
        env._update_phase()
        self.assertEqual(env.phase, "handback")
        env._update_safety_and_metrics()
        self.assertEqual(env.collision_phase, "takeover")
        env.reset(seed=100)
        self.assertIsNone(env.collision_phase)
        self.assertIsNone(env.collision_time)


class ReactivePolicyTests(unittest.TestCase):
    @staticmethod
    def observations():
        obs = np.zeros((4, 41), dtype=np.float32)
        for start in (12, 19, 26):
            obs[:, start:start + 3] = [5, 0, 0]
        return obs

    def test_each_actor_uses_only_its_own_relative_position_fields(self):
        obs = self.observations()
        obs[0, 12:15] = [0.6, 0, 0]
        expected = reactive_actions(obs)
        np.testing.assert_array_equal(expected[0], reactive_actions(obs[0]))
        changed = obs.copy()
        changed[1:] = 17
        ignored = np.ones(41, dtype=bool)
        for start in (12, 19, 26):
            ignored[start:start + 3] = False
        changed[0, ignored] = np.nan
        np.testing.assert_array_equal(reactive_actions(changed)[0], expected[0])

    def test_repulsion_formula_symmetry_and_zero_outside_range(self):
        obs = self.observations()
        obs[0, 12:15] = [0.6, 0, 0]
        obs[1, 12:15] = [-0.6, 0, 0]
        actions = reactive_actions(obs)
        np.testing.assert_allclose(actions[0], [-0.75, 0, 0], atol=1e-6)
        np.testing.assert_allclose(actions[1], [0.75, 0, 0], atol=1e-6)
        np.testing.assert_array_equal(actions[2:], np.zeros((2, 3)))

    def test_superposition_clipping_and_coincident_position(self):
        obs = self.observations()
        obs[0, 12:15] = [0.1, 0, 0]
        obs[0, 19:22] = [0, -0.1, 0]
        obs[0, 26:29] = [0, 0, 0]
        np.testing.assert_array_equal(reactive_actions(obs)[0], [-1, 1, 0])
        self.assertEqual(reactive_actions(obs).dtype, np.float32)
        with self.assertRaises(ValueError):
            reactive_actions(np.zeros((4, 40)))
        with self.assertRaises(ValueError):
            reactive_actions(obs, residual_velocity=0)


if __name__ == "__main__":
    unittest.main()
