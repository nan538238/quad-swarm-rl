"""Focused CP-C2-01 implementation checks; no scientific policy training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from swarm_rl.takeover.geometry import GeometryConfig, GeometryTakeoverEnv
from swarm_rl.takeover.policy import (TeamPolicy, actor_mask, generalized_advantages,
                                      load_policy, observation_scale, ppo_actor_loss,
                                      squashed_log_prob, state_digest, update_policy)
from swarm_rl.takeover.policy_train import (analytic_cost, auxiliary_reward,
    failure_inclusive_recovery_cost, reference_action, terminal_mission_reward,
    training_scenario, validate_specification)
from swarm_rl.takeover.recovery_diagnose import snapshot_digest


def specification():
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/takeover_e0004.json').read_text())


class PolicyMathematicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_gae_handcomputed_terminal_and_rollout_bootstrap(self):
        # First episode ends at t=1. Last transition belongs to a fresh episode
        # which continues past the rollout boundary, and must bootstrap value=4.
        advantage, returns = generalized_advantages(
            [1, 2, 3], [0.5, 1, 2], [False, True, False], 4, gamma=0.9, gae_lambda=0.8)
        np.testing.assert_allclose(advantage, [2.12, 1, 4.6], rtol=1e-6)
        np.testing.assert_allclose(returns, [2.62, 2, 6.6], rtol=1e-6)
        # An actual 24s time limit has the same end flag as collision.
        advantage, returns = generalized_advantages([3], [2], [True], 999,
                                                    gamma=0.9, gae_lambda=0.8)
        np.testing.assert_array_equal(advantage, [1])
        np.testing.assert_array_equal(returns, [3])

    def test_stable_tanh_logprob_and_exact_identity_ratio(self):
        distribution = torch.distributions.Normal(torch.zeros((2, 3)), torch.ones((2, 3)))
        latent = torch.tensor([[0.1, -0.2, 0.3], [30.0, -30.0, 20.0]])
        log_prob = squashed_log_prob(distribution, latent)
        self.assertTrue(torch.isfinite(log_prob).all())
        ordinary = (distribution.log_prob(latent[0]) - torch.log1p(-latent[0].tanh().square())).sum(-1)
        torch.testing.assert_close(log_prob[0], ordinary[0])
        torch.testing.assert_close((log_prob - squashed_log_prob(distribution, latent)).exp(),
                                   torch.ones(2), rtol=0, atol=0)
        self.assertEqual(float(latent[1, 0].tanh()), 1.0)

    def test_fixed_scaling_exact_field_indices(self):
        scale = observation_scale()
        for start in (0, 9, 15, 22, 29, 38):
            np.testing.assert_array_equal(scale[start:start+3], np.full(3, 1.2, dtype=np.float32))
        for start in (3, 6, 12, 19, 26):
            np.testing.assert_array_equal(scale[start:start+3], np.full(3, 3.0))
        self.assertEqual(scale[34], 5)
        np.testing.assert_array_equal(scale[[18, 25, 32, 33, 35, 36, 37]], np.ones(7))
        model = TeamPolicy()
        torch.testing.assert_close(model.normalize(torch.from_numpy(scale)), torch.ones(41))
        self.assertEqual(float(model.normalize(torch.ones(41)*1e8).max()), 10.0)

    def test_row_local_actor_and_masked_actions(self):
        model = TeamPolicy()
        obs = torch.randn(4, 41)
        obs[:, 33] = 0
        obs[:, 36] = 1
        obs[1, 33] = 1
        before = model.distribution(obs).mean.detach().clone()
        changed = obs.clone()
        changed[2, :33] += 9
        after = model.distribution(changed).mean.detach()
        torch.testing.assert_close(before[[0, 1, 3]], after[[0, 1, 3]], rtol=0, atol=0)
        self.assertTrue(torch.equal(actor_mask(obs), torch.tensor([True, False, True, True])))
        actions, latent, old_log, mask, _ = model.sample(obs, torch.Generator().manual_seed(0))
        torch.testing.assert_close(actions[1], torch.zeros(3), rtol=0, atol=0)
        self.assertTrue((actions.abs() <= 1).all())
        torch.testing.assert_close(old_log, squashed_log_prob(model.distribution(obs), latent), rtol=0, atol=0)
        obs[:, 36] = 0
        actions, *_ = model.sample(obs, torch.Generator().manual_seed(0))
        torch.testing.assert_close(actions, torch.zeros((4, 3)), rtol=0, atol=0)

    def test_actor_loss_uses_individual_ratios_and_masks_gradients(self):
        log_probs = torch.zeros((2, 4), requires_grad=True)
        mask = torch.tensor([[True, False, True, True], [False, False, False, False]])
        advantages = torch.tensor([2.0, 999.0])
        loss, stats = ppo_actor_loss(log_probs, log_probs.detach().clone(), advantages, mask)
        self.assertAlmostEqual(float(loss), -2.0)
        self.assertEqual(stats['approx_kl'], 0)
        loss.backward()
        torch.testing.assert_close(log_probs.grad[~mask], torch.zeros(5), rtol=0, atol=0)
        torch.testing.assert_close(log_probs.grad[mask], torch.full((3,), -2/3))
        # One agent doubles its ratio: clipping applies separately, not jointly.
        changed = torch.tensor([[np.log(2), 0, 0, 0]], dtype=torch.float32)
        loss, _ = ppo_actor_loss(changed, torch.zeros_like(changed), torch.ones(1),
                                 torch.tensor([[True, False, True, True]]))
        self.assertAlmostEqual(float(loss), -(1.2+1+1)/3, places=6)

    def test_no_actor_update_without_active_rows_but_critic_updates(self):
        model = TeamPolicy()
        obs = np.zeros((4, 4, 41), dtype=np.float32)
        before = {key: value.detach().clone() for key, value in model.named_parameters()}
        rollout = {'observations': obs, 'latents': np.zeros((4, 4, 3)),
                   'log_probs': np.zeros((4, 4)), 'masks': np.zeros((4, 4), dtype=bool),
                   'rewards': [1., 2., 3., 4.], 'values': [0., 0., 0., 0.],
                   'episode_ends': [False, False, False, True], 'bootstrap': 0.0}
        report = update_policy(model, torch.optim.Adam(model.parameters(), lr=3e-4),
            rollout, {**specification(), 'epochs': 1}, np.random.default_rng(0),
            torch.Generator().manual_seed(1000))
        for key, value in model.named_parameters():
            if not key.startswith('critic'):
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before[key]) for key, value
                            in model.named_parameters() if key.startswith('critic')))
        self.assertEqual(report['active_actor_rows'], 0)
        self.assertGreater(report['parameter_change_l2'], 0)

    def test_initial_weights_and_checkpoint_reload_are_identical(self):
        one = TeamPolicy()
        torch.manual_seed(0)
        two = TeamPolicy()
        self.assertEqual(state_digest(one), state_digest(two))
        observations = torch.randn(3, 4, 41)
        observations[..., 33] = 0
        observations[..., 36] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'state_dict': one.state_dict()}, path)
            restored, _ = load_policy(path)
            self.assertEqual(state_digest(one), state_digest(restored))
            torch.testing.assert_close(one.deterministic(observations), restored.deterministic(observations),
                                       rtol=0, atol=0)


class PolicySemanticsTests(unittest.TestCase):
    def test_complete_episode_terminal_cost_and_check_failure(self):
        metrics = {'collision': False, 'recovery_success': True,
                   'recovery_time': 3.5, 'check_completed': True}
        self.assertEqual(terminal_mission_reward(metrics, False), 0)
        self.assertEqual(terminal_mission_reward(metrics, True), -1.5)
        metrics['collision'] = True  # Collision after an earlier successful rejoin.
        self.assertEqual(failure_inclusive_recovery_cost(metrics), 10)
        self.assertEqual(terminal_mission_reward(metrics, True), -8)
        metrics.update(collision=False, check_completed=False)
        self.assertEqual(failure_inclusive_recovery_cost(metrics), 3.5)
        self.assertEqual(terminal_mission_reward(metrics, True), -3.5)
        metrics.update(recovery_success=False, recovery_time=None)
        self.assertEqual(failure_inclusive_recovery_cost(metrics), 10)

    def test_poststep_analytic_signal_collision_override_and_phase_gate(self):
        obs = np.zeros((4, 41), dtype=np.float32)
        obs[0, 3] = 0.6
        obs[1, 0] = 0.8
        self.assertAlmostEqual(analytic_cost(obs), 2.0, delta=1e-6)
        env = SimpleNamespace(collision=False, config=SimpleNamespace(control_dt=0.1))
        aux, label = auxiliary_reward('analytic', 'takeover', obs, env, None, specification())
        self.assertAlmostEqual(aux, -0.1/4.5 * 2)
        self.assertIsNone(label)
        self.assertEqual(auxiliary_reward('analytic', 'patrol', obs, env, None, specification())[0], 0)
        env.collision = True
        self.assertAlmostEqual(auxiliary_reward('analytic', 'takeover', obs, env, None,
                                                specification())[0], -0.1/4.5 * 10)

    def test_sparse_first_handback_label_restores_full_state(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(0, 0),
                                                takeover_duration_range=(0.1, 0.1)))
        env.reset(seed=300)
        obs, *_ = env.step(env.reference_actions('nominal'))
        self.assertEqual(env.phase, 'handback')
        before = snapshot_digest(env)
        reward, label = auxiliary_reward('sparse', 'takeover', obs, env, None, specification())
        self.assertEqual(snapshot_digest(env), before)
        self.assertEqual(reward, -label['cost_seconds'])
        self.assertGreater(label['simulated_control_steps'], 0)
        self.assertGreater(label['simulated_physics_steps'], 0)
        self.assertTrue(label['snapshot_restored_exactly'])
        self.assertEqual(auxiliary_reward('sparse', 'handback', obs, env, None, specification()), (0, None))
        env.collision = env.terminated = True
        with patch('swarm_rl.takeover.policy_train.label_snapshot', side_effect=AssertionError('must not label')):
            self.assertEqual(auxiliary_reward('sparse', 'takeover', obs, env, None, specification()), (0, None))

    def test_sparse_collision_overrides_recovery_in_last_label_interval(self):
        env = SimpleNamespace(phase='handback', terminated=False, truncated=False, collision=False,
                              config=SimpleNamespace(control_dt=0.1, physics_dt=0.02))
        label = {'collision': True, 'recovery_success': True, 'recovery_time': 3.5,
                 'cost_seconds': 3.5, 'simulated_physics_steps': 177}
        with patch('swarm_rl.takeover.policy_train.label_snapshot', return_value=(None, label)):
            reward, result = auxiliary_reward('sparse', 'takeover', np.zeros((4, 41)),
                                              env, None, specification())
        self.assertEqual(reward, -10.0)
        self.assertEqual(result['cost_seconds'], 10.0)
        self.assertEqual(result['simulated_control_steps'], 36)

    def test_no_future_fields_and_reference_phase_gating(self):
        env = GeometryTakeoverEnv(GeometryConfig(takeover_start_range=(0, 0)))
        obs, _ = env.reset(seed=300)
        model = TeamPolicy()
        before = model.deterministic(torch.from_numpy(obs)).detach()
        env._operator_target += [100, 100, 100]
        env._takeover_end += 100
        torch.testing.assert_close(before, model.deterministic(torch.from_numpy(env.observe())), rtol=0, atol=0)
        env.force_handback()
        obs = env.observe()
        for policy in ('nominal', 'wait_takeover', 'reactive_takeover'):
            np.testing.assert_array_equal(reference_action(policy, env, obs, specification()), np.zeros((4, 3)))

    def test_fixed_schedule_and_split_guards(self):
        spec = specification()
        validate_specification(spec)
        self.assertEqual([training_scenario(i) for i in range(9)],
                         [(300, angle) for angle in range(0, 360, 45)] + [(301, 0)])
        self.assertEqual(training_scenario(5600), (300, 0))
        for key, value in [('validation_scenario_seeds', [20000]),
                           ('training_seed_range', [10020, 10029]), ('training_steps', 32767)]:
            with self.assertRaises(ValueError):
                validate_specification({**spec, key: value})


if __name__ == '__main__':
    unittest.main()
