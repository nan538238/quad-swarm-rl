"""Engineering-only CP2 seed, checkpoint, and split checks."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from swarm_rl.takeover import policy_full as full, policy_train as core
from swarm_rl.takeover.policy import TeamPolicy, load_policy, state_digest

ROOT = Path(__file__).resolve().parents[1]


def specification():
    return json.loads((ROOT / 'configs/takeover_e0006.json').read_text())


class FullProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_seeds_variants_have_identical_initialization_and_correct_metadata(self):
        digests = []
        for seed in (0, 1, 2):
            initial = []
            for variant in ('none', 'analytic', 'dense', 'sparse'):
                spec = dict(specification(), torch_seed=seed)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    with self.assertRaises(TimeoutError):
                        core.train(spec, variant, output, None, time.monotonic() - 1)
                    record = json.loads((output / 'initialization.json').read_text())
                    self.assertEqual(record['rng_seeds'], {'initial_weights': seed,
                        'action': seed, 'shuffle': seed, 'entropy': 1000 + seed})
                    initial.append(record['state_sha256'])
            self.assertEqual(len(set(initial)), 1)
            digests.append(initial[0])
        self.assertEqual(len(set(digests)), 3)

    def test_milestone_preserves_rng_parameters_and_reloads_exactly(self):
        model = TeamPolicy()
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        before = state_digest(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            full.save_milestone(model, {'live_control_steps': 32768, 'updates': 16},
                                path, specification(), 'none')
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
            self.assertEqual(state_digest(model), before)
            loaded, saved = load_policy(path / 'checkpoints/step-32768.pt')
            self.assertEqual(state_digest(loaded), before)
            self.assertEqual(saved['counts']['updates'], 16)

    def test_checkpoint_capture_leaves_actual_training_trajectory_unchanged(self):
        # Bounded integration test, two short PPO updates; never scientific evidence.
        spec = dict(specification(), torch_seed=1, training_steps=512,
                    rollout_timesteps=256, checkpoint_steps=[256, 512])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('plain', 'captured'):
                (root / name).mkdir()
                hook = (lambda model, counts: full.save_milestone(
                    model, counts, root / name, spec, 'none')) if name == 'captured' else None
                core.train(spec, 'none', root / name, None, time.monotonic() + 60, hook)
            plain = torch.load(root / 'plain/model.pt', weights_only=False)
            captured = torch.load(root / 'captured/model.pt', weights_only=False)
            self.assertEqual(plain['final_state_sha256'], captured['final_state_sha256'])
            for key in ('action_rng_state', 'entropy_rng_state'):
                self.assertTrue(torch.equal(plain[key], captured[key]))
            self.assertEqual(plain['shuffle_rng_state'], captured['shuffle_rng_state'])
            import gzip
            self.assertEqual(gzip.open(root / 'plain/training_transitions.jsonl.gz').read(),
                             gzip.open(root / 'captured/training_transitions.jsonl.gz').read())

    def test_protocol_rejects_seed_split_mechanism_or_rule_changes(self):
        spec = specification()
        full.validate_specification(spec, 'none', 0, True)
        for seed in (1, 2):
            full.validate_specification(spec, 'dense', seed, False)
        for key, value in [('final_test_scenario_seeds', [10040]), ('training_steps', 32768),
                           ('learning_rate', .001), ('torch_seed', 0)]:
            with self.assertRaises(ValueError):
                full.validate_specification(dict(spec, **{key: value}), 'none', 0, True)
        for variant, seed, rules in [('none', 3, False), ('none', 0, False),
                                      ('dense', 1, True), ('sparse', 1, False)]:
            with self.assertRaises(ValueError):
                full.validate_specification(spec, variant, seed, rules)

    def test_split_guard_prevents_final_evaluation_of_intermediate_snapshot(self):
        with patch.object(core, 'evaluate') as evaluate:
            with self.assertRaises(ValueError):
                full.evaluate_split(specification(), 'none', None, Path('/unused'),
                                    0, 'final_test', 32768)
            evaluate.assert_not_called()

    def test_split_labels_and_seed_lists_are_explicit(self):
        for split, seeds in [('development', list(range(10040, 10045))),
                             ('final_test', list(range(20000, 20020)))]:
            rows = [{'status': 'complete', 'split': split}] * (len(seeds) * 8)
            with tempfile.TemporaryDirectory() as directory, patch.object(core, 'evaluate',
                    return_value=(rows, {})) as evaluate, patch.object(core, 'summarize', return_value={}):
                full.evaluate_split(specification(), 'none', None, Path(directory) / 'out',
                                    123, split, 131072)
                self.assertEqual(evaluate.call_args.kwargs, {'split': split, 'scenario_seeds': seeds})


if __name__ == '__main__':
    unittest.main()
