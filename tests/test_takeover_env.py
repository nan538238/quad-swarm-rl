"""Semantic/physics checks for E0001; no claims about method effectiveness."""

from dataclasses import replace
import unittest

import numpy as np

from swarm_rl.takeover import TakeoverConfig, TakeoverPatrolEnv
from gym_art.quadrotor_multi.quadrotor_dynamics import QuadrotorDynamics


class TakeoverEnvironmentTests(unittest.TestCase):
    def test_seeded_replay_and_snapshot_include_motor_state_and_rng(self):
        config = TakeoverConfig(takeover_start_range=(0.5, 0.5),
                                takeover_duration_range=(1.5, 1.5))
        a, b = TakeoverPatrolEnv(config), TakeoverPatrolEnv(config)
        np.testing.assert_array_equal(a.reset(seed=3)[0], b.reset(seed=3)[0])
        actions = np.random.default_rng(8).uniform(-0.1, 0.1, (15, 4, 3))
        for action in actions[:8]:
            np.testing.assert_array_equal(a.step(action)[0], b.step(action)[0])
        saved = a.snapshot()
        filters = [d.thrust_rot_damp.copy() for d in a.dynamics]
        expected = [a.step(action)[0] for action in actions[8:]]
        expected_metrics = a.metrics()
        expected_rng = a.rng.random(5)
        a.restore(saved)
        for dyn, filt in zip(a.dynamics, filters):
            np.testing.assert_array_equal(dyn.thrust_rot_damp, filt)
        for action, obs in zip(actions[8:], expected):
            np.testing.assert_array_equal(a.step(action)[0], obs)
        self.assertEqual(a.metrics(), expected_metrics)
        np.testing.assert_array_equal(a.rng.random(5), expected_rng)

    def test_only_taken_over_uav_action_is_overridden(self):
        config = TakeoverConfig(takeover_start_range=(0, 0),
                                takeover_duration_range=(2, 2))
        a, b = TakeoverPatrolEnv(config), TakeoverPatrolEnv(config)
        a.reset(seed=0)
        b.reset(seed=0)
        action = np.full((4, 3), 0.2)
        alternative = action.copy()
        alternative[a.takeover_agent] = -0.8
        _, _, _, _, info = a.step(action)
        b.step(alternative)
        mask = info["override_mask"]
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(mask[a.takeover_agent])
        np.testing.assert_array_equal(info["requested_velocity"][~mask],
                                      info["executed_velocity"][~mask])
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.velocities, b.velocities)
        self.assertFalse(np.array_equal(info["requested_velocity"][mask],
                                        info["executed_velocity"][mask]))

    def test_observation_has_no_future_schedule_or_hidden_target(self):
        env = TakeoverPatrolEnv(TakeoverConfig(takeover_start_range=(0, 0)))
        obs, _ = env.reset(seed=4)
        self.assertEqual(obs.shape, (4, 41))
        env._takeover_end += 50
        env._operator_target += np.array([10, -10, 3])
        np.testing.assert_array_equal(obs, env.observe())
        env.step(np.zeros((4, 3)))
        self.assertTrue(np.isfinite(env.observe()).all())

    def test_handback_flag_and_nominal_control_return(self):
        env = TakeoverPatrolEnv(TakeoverConfig(takeover_start_range=(0, 0),
                                takeover_duration_range=(0.2, 0.2)))
        env.reset(seed=1)
        env.step(np.zeros((4, 3)))
        self.assertEqual(env.phase, "takeover")
        env.step(np.zeros((4, 3)))
        self.assertEqual(env.phase, "handback")
        self.assertEqual(env.takeover_flags.sum(), 0)
        self.assertAlmostEqual(env.handover_time, 0.2)
        info = env.step(np.full((4, 3), 0.1))[-1]
        self.assertFalse(info["override_mask"].any())
        np.testing.assert_array_equal(info["requested_velocity"], info["executed_velocity"])

    def test_no_takeover_patrol_uses_stable_six_dof_physics(self):
        env = TakeoverPatrolEnv(TakeoverConfig(takeover_enabled=False,
                                              episode_seconds=8))
        env.reset(seed=2)
        self.assertIsInstance(env.dynamics[0], QuadrotorDynamics)
        done = truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, _ = env.step(env.reference_actions())
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
        self.assertFalse(env.collision)
        self.assertTrue(truncated)
        self.assertGreater(env.metrics()["patrol_progress"], 2.0)
        self.assertLess(np.max(np.abs(env.positions[:, 2] - env.config.altitude)), 0.2)
        self.assertLess(np.max(np.linalg.norm(env.positions - env.slot_targets, axis=1)), 0.3)
        for dyn in env.dynamics:
            np.testing.assert_allclose(dyn.rot.T @ dyn.rot, np.eye(3), atol=1e-5)

    def test_recovery_requires_continuous_hold_and_keeps_failure(self):
        env = TakeoverPatrolEnv(TakeoverConfig(recovery_hold_seconds=0.06))
        env.reset(seed=0)
        env.force_handback()
        for _ in range(2):
            env.physics_steps += 1
            env._update_recovery()
        self.assertFalse(env.recovery_success)
        env.dynamics[0].vel[0] = 2.0
        env.physics_steps += 1
        env._update_recovery()
        self.assertEqual(env._recovery_hold, 0.0)
        for _ in range(3):
            env.physics_steps += 1
            for dyn, target in zip(env.dynamics, env.slot_targets):
                dyn.pos = target.copy()
                dyn.vel = env.nominal_velocity.copy()
            env._update_recovery()
        self.assertTrue(env.recovery_success)
        self.assertAlmostEqual(env.recovery_time, 0.12)

        env.reset(seed=0)
        env.force_handback()
        env.dynamics[0].pos += 10
        env.physics_steps = round(env.config.recovery_timeout / env.config.physics_dt)
        env._update_recovery()
        metrics = env.metrics()
        self.assertFalse(metrics["recovery_success"])
        self.assertIsNone(metrics["recovery_time"])
        self.assertTrue(metrics["recovery_censored"])
        self.assertTrue(metrics["recovery_decided"])
        self.assertEqual(metrics["recovery_time_with_timeout"], env.config.recovery_timeout)

    def test_counterfactual_rollout_restores_live_state_exactly(self):
        env = TakeoverPatrolEnv(TakeoverConfig(takeover_start_range=(0, 0)))
        env.reset(seed=2)
        for _ in range(3):
            env.step(np.zeros((4, 3)))
        saved = env.snapshot()
        before = env.observe().copy()
        metrics = env.metrics()
        label = env.evaluate_immediate_handback(max_seconds=0.3)
        self.assertGreater(label["simulated_physics_steps"], 0)
        self.assertFalse(label["protocol_horizon_complete"])
        np.testing.assert_array_equal(before, env.observe())
        self.assertEqual(metrics, env.metrics())
        after = env.step(np.zeros((4, 3)))[0]
        env.restore(saved)
        np.testing.assert_array_equal(after, env.step(np.zeros((4, 3)))[0])

    def test_collision_is_terminal_and_does_not_become_recovery_success(self):
        env = TakeoverPatrolEnv()
        env.reset(seed=0)
        env.force_handback()
        env.dynamics[0].pos = env.dynamics[1].pos.copy()
        _, _, done, _, info = env.step(np.zeros((4, 3)))
        self.assertTrue(done)
        self.assertTrue(info["collision"])
        self.assertFalse(info["recovery_success"])
        self.assertTrue(info["recovery_censored"])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros((4, 3)))

    def test_stationary_team_does_not_receive_patrol_progress(self):
        env = TakeoverPatrolEnv()
        env.reset(seed=0)
        env.physics_steps = 100
        metrics = env.metrics()
        self.assertEqual(metrics["patrol_progress"], 0.0)
        self.assertGreater(metrics["patrol_progress_loss"], 0)

    def test_invalid_action_and_timestep_rejected(self):
        with self.assertRaises(ValueError):
            TakeoverConfig(control_dt=0.11)
        env = TakeoverPatrolEnv()
        env.reset(seed=0)
        for action in (np.zeros((4, 4)), np.full((4, 3), np.nan)):
            with self.assertRaises(ValueError):
                env.step(action)


if __name__ == "__main__":
    unittest.main()
