"""Authority, causal information and manual-control invariants."""
from io import StringIO
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from swarm_rl.takeover.adaptation import (AUTHORITY_COLUMNS, AdaptationConfig,
    AdaptationEnv, AdaptationPolicy, ManualTakeoverEnv, ObservationHistory, make_env)
from swarm_rl.takeover.adaptation_train import autonomous_mask, evaluation_case
from swarm_rl.takeover.keyboard import Session


class AdaptationTests(unittest.TestCase):
    def test_history_causal_row_local_and_reset(self):
        history = ObservationHistory('history', 4)
        first = np.ones((4, 41), dtype=np.float32)
        obs = history.reset(first)
        np.testing.assert_array_equal(obs[:, 41:], 0)
        second = first * 2
        second[2] = 3
        obs = history.push(second)
        np.testing.assert_array_equal(obs[:, :41], second)
        np.testing.assert_array_equal(obs[:, 41:82], first)
        np.testing.assert_array_equal(obs[:, 82:], 0)
        np.testing.assert_array_equal(history.reset(first)[:, 41:], 0)

    def test_blind_hides_all_explicit_authority_and_current_only_variants(self):
        obs = np.ones((4, 41), dtype=np.float32)
        blind = ObservationHistory('blind').reset(obs)
        np.testing.assert_array_equal(blind[:, AUTHORITY_COLUMNS], 0)
        np.testing.assert_array_equal(blind[:, :18], 1)
        flag = ObservationHistory('flag')
        flag.reset(obs)
        result = flag.push(obs * 2)
        np.testing.assert_array_equal(result[:, :41], 2)
        np.testing.assert_array_equal(result[:, 41:], 0)

    def test_same_parameter_budget_and_row_local_actor(self):
        torch.manual_seed(1)
        model = AdaptationPolicy()
        obs = torch.randn(4, 164)
        before = model.distribution(obs).mean.detach()
        obs[2] += 7
        after = model.distribution(obs).mean.detach()
        torch.testing.assert_close(before[[0, 1, 3]], after[[0, 1, 3]], rtol=0, atol=0)
        counts = []
        for variant in ('blind', 'flag', 'history'):
            counts.append(sum(p.numel() for p in AdaptationPolicy().parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_control_continues_after_handback_and_override_mask(self):
        env = AdaptationEnv(AdaptationConfig(takeover_start_range=(0, 0),
            takeover_duration_range=(0.1, 0.1)))
        raw, _ = env.reset(seed=9)
        mask = autonomous_mask(env)
        self.assertEqual(int(mask.sum()), 3)
        model = AdaptationPolicy()
        obs = torch.from_numpy(ObservationHistory('blind').reset(raw))
        actions, *_ = model.sample(obs, mask, torch.Generator().manual_seed(0))
        np.testing.assert_array_equal(actions.detach().numpy()[env.takeover_agent], 0)
        env.step(actions.detach().numpy())
        self.assertEqual(env.phase, 'handback')
        self.assertEqual(int(autonomous_mask(env).sum()), 4)

    def test_command_delay_held_per_control_tick_and_seed_reproducibility(self):
        config = AdaptationConfig(takeover_start_range=(0, 0), command_delay_steps=2,
                                  command_noise_std=0.1)
        env, other = AdaptationEnv(config), AdaptationEnv(config)
        env.reset(seed=10)
        other.reset(seed=10)
        for step in range(4):
            a, b = env.operator_command(), other.operator_command()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, env.operator_command())
            if step < 2:
                np.testing.assert_array_equal(a, 0)
            else:
                self.assertGreater(np.linalg.norm(a), 0)
            env.control_steps += 1
            other.control_steps += 1

    def test_manual_only_switches_on_events_and_executes_human_velocity(self):
        env = ManualTakeoverEnv(AdaptationConfig(takeover_start_range=(0, 0),
            takeover_duration_range=(0.1, 0.1)))
        env.reset(seed=11)
        self.assertEqual(env.phase, 'patrol')
        env.begin_takeover()
        env.manual_velocity = np.array([0, 0.4, 0.1])
        env.step(np.zeros((4, 3)))
        env.step(np.zeros((4, 3)))
        self.assertEqual(env.phase, 'takeover')
        np.testing.assert_allclose(env.last_executed_velocity[env.takeover_agent], [0, 0.4, 0.1])
        env.force_handback()
        self.assertEqual(env.phase, 'handback')
        self.assertEqual(int(autonomous_mask(env).sum()), 4)
        with self.assertRaises(ValueError):
            env.force_handback()

    def test_keyboard_history_event_does_not_erase_prior_frames_and_abort_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(SimpleNamespace(output=str(Path(directory) / 'session'),
                checkpoint=None, geometry='crossing', seed=60000))
            raw = session.env.observe()
            session.history = ObservationHistory('history', 4)
            session.obs = session.history.reset(raw)
            session.obs = session.history.push(raw + 0.01)
            session.input({'event': 'takeover', 'keys': []})
            np.testing.assert_array_equal(session.obs[:, 41:82], raw)
            self.assertEqual(session.env.phase, 'takeover')
            session.input({'event': 'reset', 'keys': []})
            import json
            aborted = json.loads((session.output / 'episode-0000.json').read_text())
            self.assertFalse(aborted['complete_handoff_cycle'])
            self.assertFalse(aborted['episode_ended'])
            self.assertEqual(aborted['reason'], 'user_reset')
            session.events.close()
            session.trajectory.close()


if __name__ == '__main__':
    unittest.main()
