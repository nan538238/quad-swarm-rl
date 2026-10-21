"""E0003 semantic checks; tiny synthetic fit is an engineering test only."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from swarm_rl.takeover.geometry import GeometryConfig, GeometryTakeoverEnv
from swarm_rl.takeover.recovery_diagnose import (collector_actions, label_snapshot,
                                                snapshot_digest, validate_specification)
from swarm_rl.takeover.recovery_model import (RecoveryCostModel, errors, fit_normalizer,
                                             fit_ridge, geometry_features, joint_features,
                                             load_predictor, predict, train_predictor)


class RecoveryFeaturesTests(unittest.TestCase):
    def test_joint_features_use_only_current_observation(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(0, 0)))
        obs, _ = env.reset(seed=200)
        features = joint_features(obs)
        self.assertEqual(features.shape, (164,))
        env._operator_target += [50, 50, 50]
        env._takeover_end += 100
        np.testing.assert_array_equal(features, joint_features(env.observe()))
        with self.assertRaises(ValueError):
            joint_features(np.zeros((4, 42)))

    def test_geometry_features_exact_fields_and_floor(self):
        obs = np.zeros((2, 4, 41))
        obs[..., 9] = 0.3
        obs[..., 0] = 0.3
        obs[:, 2, 3:6] = [3, 4, 0]
        obs[:, 1, 0:3] = [0.3, 0, 2]
        for start in (12, 19, 26):
            obs[..., start:start+3] = [2, 0, 0]
        expected = np.array([[1, 5, 2, 0.5], [1, 5, 2, 0.5]])
        np.testing.assert_allclose(geometry_features(obs), expected)
        obs[0, 0, 12:15] = 0
        self.assertEqual(geometry_features(obs)[0, 3], 10)
        obs[..., 34:] = 999  # Not geometry features.
        self.assertEqual(geometry_features(obs)[1, 3], 0.5)

    def test_normalization_and_ridge_fit_training_only(self):
        train = np.stack([np.zeros(164), np.ones(164) * 2])
        mean, scale = fit_normalizer(train)
        np.testing.assert_array_equal(mean, np.ones(164))
        np.testing.assert_array_equal(scale, np.ones(164))
        val = np.ones((3, 164)) * 1000
        self.assertTrue(np.all((val-mean)/scale == 999))
        geometry = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0]])
        target = np.array([1., 3., 5.])
        coefficients = fit_ridge(geometry, target)
        self.assertAlmostEqual(coefficients[1], 2.0, places=2)
        self.assertLess(np.max(np.abs(geometry @ coefficients - target)), 0.002)
        self.assertEqual(errors([], [])['count'], 0)

    def test_disjoint_split_rejects_test_and_overlap(self):
        spec = {'training_scenario_seeds': [200], 'validation_scenario_seeds': [10000],
                'angles_degrees': list(range(0, 360, 45)),
                'checkpoint_elapsed_seconds': [1.0, 3.0]}
        validate_specification(spec)
        for replacement in ([200], [20000]):
            with self.assertRaises(ValueError):
                validate_specification({**spec, 'validation_scenario_seeds': replacement})


class RecoveryLabelsTests(unittest.TestCase):
    def test_full_label_restores_geometry_state_and_ignores_future(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(0, 0),
                                                detour_angle_degrees=90))
        env.reset(seed=201)
        for _ in range(3):
            env.step(env.reference_actions('nominal'))
        before = snapshot_digest(env)
        obs, label = label_snapshot(env)
        self.assertEqual(snapshot_digest(env), before)
        self.assertEqual(obs.shape, (4, 41))
        self.assertEqual(label['rollout_horizon'], 10.0)
        self.assertTrue(label['protocol_horizon_complete'])
        self.assertGreater(label['simulated_physics_steps'], 0)
        self.assertIsNone(env.collision_phase)
        self.assertIsNone(env.collision_time)
        env._operator_target += [20, -10, 4]
        env._takeover_end += 100
        changed_obs, changed_label = label_snapshot(env)
        np.testing.assert_array_equal(obs, changed_obs)
        self.assertEqual(label, changed_label)

    def test_noise_is_independent_of_environment_rng_and_patrol(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(1, 1)))
        obs, _ = env.reset(seed=200)
        spec = {'reactive_gain': 1.5, 'reactive_range_m': 0.9, 'action_noise_std': 0.15}
        action_rng = np.random.default_rng(123)
        before = snapshot_digest(env)
        np.testing.assert_array_equal(collector_actions(env, obs, 200, action_rng, spec),
                                      env.reference_actions('nominal'))
        for _ in range(10):
            obs, *_ = env.step(env.reference_actions('nominal'))
        before = snapshot_digest(env)
        one = collector_actions(env, obs, 200, action_rng, spec)
        self.assertEqual(snapshot_digest(env), before)
        two = collector_actions(env, obs, 200, np.random.default_rng(123), spec)
        np.testing.assert_array_equal(one, two)
        self.assertTrue(np.any(one != 0))

    def test_counterfactual_collision_is_timeout_cost_and_restores_subclass(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(0, 0)))
        env.reset(seed=203)
        # A deliberately imminent contact exercises failed-label accounting.
        env.dynamics[1].pos = env.dynamics[0].pos.copy()
        before = snapshot_digest(env)
        _, label = label_snapshot(env)
        self.assertTrue(label['collision'])
        self.assertFalse(label['recovery_success'])
        self.assertEqual(label['cost_seconds'], 10.0)
        self.assertGreater(label['simulated_physics_steps'], 0)
        self.assertEqual(snapshot_digest(env), before)
        self.assertIsNone(env.collision_phase)
        self.assertIsNone(env.collision_time)
        self.assertFalse(env.collision)


class RecoveryModelTests(unittest.TestCase):
    def test_output_bounds_and_checkpoint_reload(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1)
        train = rng.normal(size=(12, 164)).astype(np.float32)
        val = rng.normal(size=(4, 164)).astype(np.float32)
        model = RecoveryCostModel()
        outputs = predict(model, train * 1000)
        self.assertTrue(np.all((outputs >= 0) & (outputs <= 10)))
        spec = {'torch_seed': 0, 'device': 'cpu', 'hidden_sizes': [64, 64],
                'learning_rate': 0.001, 'epochs': 3, 'batch_size': 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            trained, report = train_predictor(train, np.linspace(0, 10, 12), val,
                                               np.linspace(0, 10, 4), spec, path)
            reloaded, checkpoint = load_predictor(path)
            np.testing.assert_array_equal(predict(trained, val), predict(reloaded, val))
            np.testing.assert_allclose(checkpoint['state_dict']['mean'].numpy(),
                                       train.astype(np.float64).mean(axis=0), atol=1e-7)
            self.assertEqual(report['checkpoint_reload_max_absolute_error'], 0)
            self.assertEqual(report['selected_epoch'], min(report['epoch_history'],
                             key=lambda row: row['validation_mse'])['epoch'])


if __name__ == '__main__':
    unittest.main()
