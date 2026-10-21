import unittest
import numpy as np

from swarm_rl.delay_awareness.core import DelayedChannel, RouteEnv, reactive, predictive


class DelayAwarenessTests(unittest.TestCase):
    def test_new_task_has_no_takeover_or_privileged_actor_observation(self):
        env = RouteEnv()
        observation, _ = env.reset(seed=71000)
        self.assertIsNone(observation)
        self.assertFalse(env.config.takeover_enabled)
        np.testing.assert_allclose(np.linalg.norm(env.goals - env.positions, axis=1), 6)
        for _ in range(5):
            env.step(np.zeros((4, 3)))
        self.assertEqual(env.phase, 'patrol')
        self.assertFalse(env.takeover_flags.any())

    def test_zero_delay_all_estimators_identical(self):
        env = RouteEnv()
        env.reset(seed=71000)
        channel = DelayedChannel(0)
        for _ in range(5):
            raw, _ = channel.observation(env, 'raw')
            cv, _ = channel.observation(env, 'cv')
            oracle, _ = channel.observation(env, 'oracle')
            np.testing.assert_array_equal(raw, cv)
            np.testing.assert_array_equal(raw, oracle)
            np.testing.assert_array_equal(reactive(raw, env.config), reactive(oracle, env.config))
            env.step(np.zeros((4, 3)))

    def test_delay_uses_sender_old_position_and_current_ego_position(self):
        env = RouteEnv()
        env.reset(seed=71001)
        channel = DelayedChannel(2)
        initial = env.positions.copy()
        channel.observation(env)
        env.step(np.zeros((4, 3)))
        env.step(np.zeros((4, 3)))
        obs, info = channel.observation(env)
        self.assertAlmostEqual(info['packet_age'], .2)
        np.testing.assert_allclose(obs[0, 12:15], initial[1] - env.positions[0], atol=1e-6)
        np.testing.assert_allclose(obs[:, :3], env.velocities, atol=1e-6)

    def test_unreceived_peer_state_mutation_cannot_change_actor_input(self):
        env = RouteEnv()
        env.reset(seed=71000)
        channel = DelayedChannel(4)
        channel.observation(env)
        env.step(np.zeros((4, 3)))
        before, _ = channel.observation(env)
        env.dynamics[1].pos += np.array([10., 5., 0.])
        env.dynamics[1].vel += np.array([2., 0., 0.])
        after, _ = channel.observation(env)
        np.testing.assert_array_equal(before[0], after[0])
        oracle, _ = channel.observation(env, 'oracle')
        self.assertFalse(np.array_equal(after[0], oracle[0]))

    def test_cv_constant_velocity_projection_is_exact(self):
        env = RouteEnv()
        env.reset(seed=71000)
        for d in env.dynamics:
            d.vel[:] = [0.5, 0.1, 0.]
        channel = DelayedChannel(4)
        channel.observation(env)
        # Synthetic constant velocity state, only for estimator identity test.
        env.physics_steps = 10
        env.control_steps = 2
        for d in env.dynamics:
            d.pos += d.vel * env.time
        raw, raw_error = channel.observation(env)
        cv, cv_error = channel.observation(env, 'cv')
        self.assertGreater(raw_error['peer_position_squared_error_sum'], 0)
        self.assertAlmostEqual(cv_error['peer_position_squared_error_sum'], 0)

    def test_rule_actions_bounded_and_row_local(self):
        env = RouteEnv()
        env.reset(seed=71000)
        obs, _ = DelayedChannel(0).observation(env)
        for controller in (reactive, predictive):
            before = controller(obs, env.config)
            modified = obs.copy()
            modified[2, 12:] += 2
            after = controller(modified, env.config)
            np.testing.assert_array_equal(before[[0, 1, 3]], after[[0, 1, 3]])
            self.assertTrue(np.isfinite(after).all())
            self.assertLessEqual(np.abs(after).max(), 1)

    def test_collision_and_timeout_retained_as_failure(self):
        env = RouteEnv()
        env.reset(seed=71000)
        env.dynamics[1].pos = env.dynamics[0].pos.copy()
        _, _, terminated, _, metrics = env.step(np.zeros((4, 3)))
        self.assertTrue(terminated)
        self.assertTrue(metrics['collision'])
        self.assertFalse(metrics['success'])
        self.assertEqual(metrics['failure_inclusive_completion_cost'], 20)


if __name__ == '__main__':
    unittest.main()
