import unittest
import numpy as np
import torch

from swarm_rl.delay_awareness.learning import History, Policy
from swarm_rl.takeover.policy import update_policy


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_finite_history_has_correct_order_and_clean_episode_boundary(self):
        history = History(8)
        first = np.ones((4, 36), dtype=np.float32)
        initial = history.push(first)
        np.testing.assert_array_equal(initial[:, :-36], 0)
        second = history.push(first * 2)
        np.testing.assert_array_equal(second[:, -72:-36], first)
        np.testing.assert_array_equal(second[:, -36:], first * 2)
        np.testing.assert_array_equal(History(8).push(first)[:, :-36], 0)

    def test_mlp_current_only_gru_uses_history_both_row_local(self):
        obs = torch.randn(4, 288)
        changed_history = obs.clone()
        changed_history[:, -72:-36] += 5
        mlp, gru = Policy('mlp'), Policy('gru')
        torch.testing.assert_close(mlp.distribution(obs).mean, mlp.distribution(changed_history).mean)
        self.assertGreater(float((gru.distribution(obs).mean - gru.distribution(changed_history).mean).abs().max()), 1e-5)
        for model in (mlp, gru):
            before = model.distribution(obs).mean.detach()
            changed = obs.clone()
            changed[2] += 5
            after = model.distribution(changed).mean.detach()
            torch.testing.assert_close(before[[0, 1, 3]], after[[0, 1, 3]])

    def test_gru_ppo_updates_recurrent_weights_with_finite_loss(self):
        model = Policy('gru')
        obs = torch.randn(8, 4, 288)
        with torch.no_grad():
            actions, latent, log_prob, value = model.sample(obs, torch.Generator().manual_seed(2))
        before = model.encoder.weight_hh_l0.detach().clone()
        rollout = {'observations': obs.numpy(), 'latents': latent.numpy(), 'log_probs': log_prob.numpy(),
            'masks': np.ones((8, 4), dtype=bool), 'rewards': [1., -2., 0., 3., 1., -1., 0., 2.],
            'values': value.numpy(), 'episode_ends': [False]*7+[True], 'bootstrap': 0.0}
        spec = {'gamma': .995, 'gae_lambda': .95, 'epochs': 1, 'minibatch_timesteps': 4,
                'clip': .2, 'value_coefficient': .5, 'entropy_coefficient': .001, 'max_grad_norm': .5}
        report = update_policy(model, torch.optim.Adam(model.parameters(), lr=3e-4), rollout,
            spec, np.random.default_rng(0), torch.Generator().manual_seed(3))
        self.assertGreater(report['parameter_change_l2'], 0)
        self.assertFalse(torch.equal(before, model.encoder.weight_hh_l0))
        self.assertTrue(torch.isfinite(model.encoder.weight_hh_l0).all())


if __name__ == '__main__':
    unittest.main()
