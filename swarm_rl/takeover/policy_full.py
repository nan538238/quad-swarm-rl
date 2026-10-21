"""CP-C2-02 orchestration only; all task and PPO mechanics are shared with CP1."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

from . import policy_train as core
from .policy import load_policy, state_digest
from .recovery_diagnose import write_json
from .recovery_model import load_predictor

MILESTONES = [32768, 65536, 98304, 131072]


def validate_specification(spec, variant, seed, include_rules):
    fixed = {'protocol_id': 'CP-C2-02', 'config_id': 'takeover-policy-cp-c2-02-v1',
             'training_steps': 131072, 'training_seeds': [0, 1, 2],
             'checkpoint_steps': MILESTONES,
             'development_scenario_seeds': list(range(10040, 10045)),
             'final_test_scenario_seeds': list(range(20000, 20020)),
             'max_wall_seconds': 1790, 'rules_max_wall_seconds': 2690}
    for key, value in fixed.items():
        if spec.get(key) != value:
            raise ValueError(f'CP-C2-02 fixed field changed: {key}')
    if seed not in (0, 1, 2) or variant not in ('none', 'analytic', 'dense', 'sparse'):
        raise ValueError('Unapproved training seed or variant')
    if spec['experiment_id'] != ('E0007' if variant == 'sparse' else 'E0006'):
        raise ValueError('Variant/Card mismatch')
    if include_rules != (variant == 'none' and seed == 0):
        raise ValueError('Rules required exactly once in none seed0')
    # Validate all unchanged CP1 fields with the existing authoritative guard.
    inherited = dict(spec, protocol_id='CP-C2-01', config_id='takeover-policy-cp-c2-01-v1',
                     experiment_id='E0004', training_steps=32768, torch_seed=0,
                     validation_scenario_seeds=list(range(10020, 10030)))
    core.validate_specification(inherited)
    if 'torch_seed' in spec or 'validation_scenario_seeds' in spec:
        raise ValueError('Seed/split must be supplied only through CP2 orchestration')


def save_milestone(model, counts, output, spec, variant):
    """Serialize after update without sampling, reloading, or evaluating."""
    step = counts['live_control_steps']
    if step not in spec['checkpoint_steps']:
        return
    path = output / 'checkpoints' / f'step-{step}.pt'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch.save({'state_dict': model.state_dict(), 'counts': dict(counts),
                'specification': spec, 'variant': variant,
                'state_sha256': state_digest(model),
                'checkpoint_selection': 'prespecified_development_only'}, path)


def evaluate_split(spec, variant, model, output, deadline, split, step):
    if split == 'development' and step in MILESTONES:
        seeds = spec['development_scenario_seeds']
    elif split == 'final_test' and step == 131072:
        seeds = spec['final_test_scenario_seeds']
    else:
        raise ValueError('Disallowed split/checkpoint pair')
    output.mkdir(parents=True, exist_ok=False)
    cases, counts = core.evaluate(spec, variant, model, output, deadline,
                                 split=split, scenario_seeds=seeds)
    expected = len(seeds) * 8 * (4 if variant == 'rules' else 1)
    if len(cases) != expected or any(case['status'] != 'complete' or case['split'] != split for case in cases):
        raise RuntimeError('Incomplete or mislabeled evaluation')
    return {'checkpoint_step': step, 'split': split, 'compute': counts,
            'summary': core.summarize(cases)}


def run(config_path, variant, seed, output, include_rules=False):
    started = time.monotonic()
    config = json.loads(config_path.read_text())
    validate_specification(config, variant, seed, include_rules)
    spec = dict(config, torch_seed=seed,
                max_wall_seconds=config['rules_max_wall_seconds'] if include_rules else config['max_wall_seconds'])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    deadline = started + spec['max_wall_seconds']
    metadata = {'status': 'initializing', 'variant': variant, 'training_seed': seed}
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'effective_config.json', spec)
    try:
        metadata = core.source_metadata(spec, config_path, output, variant)
        if metadata['git_status'].strip():
            raise RuntimeError('Scientific execution requires clean committed source')
        metadata.update(training_seed=seed, include_rules=include_rules,
                        effective_config_sha256=core.file_hash(output / 'effective_config.json'),
                        checkpoint_selection='fixed_final_step_only; no development selection',
                        evaluation_order='all training, four development snapshots, final test, optional rules',
                        limitations=['Three seeds and 64 updates do not establish convergence.',
                                     'Scripted operator, fixed geometry; PRELIMINARY_NON_PAPER.',
                                     'Concurrent CPU wall time does not support timing superiority.',
                                     'Auxiliary timing/magnitude differ; common-success lag comparisons required.'])
        write_json(output / 'metadata.json', metadata)
        predictor = None
        if variant == 'dense':
            path = Path(__file__).resolve().parents[2] / spec['dense_predictor_path']
            if core.file_hash(path) != core.PREDICTOR_SHA256:
                raise RuntimeError('Frozen predictor identity mismatch')
            shutil.copyfile(path, output / 'frozen_predictor.pt')
            predictor, _ = load_predictor(output / 'frozen_predictor.pt', device='cpu')
            predictor.requires_grad_(False)
            predictor_digest = state_digest(predictor)
            metadata.update(predictor_sha256=core.PREDICTOR_SHA256, predictor_source=str(path))
        model, training = core.train(spec, variant, output, predictor, deadline,
            checkpoint_callback=lambda model, counts: save_milestone(model, counts, output, spec, variant))
        write_json(output / 'training_summary.json', training)
        if training['live_control_steps'] != 131072 or training['updates'] != 64:
            raise RuntimeError('Wrong fixed training budget')
        if predictor is not None:
            if (state_digest(predictor) != predictor_digest or
                core.file_hash(path) != core.PREDICTOR_SHA256 or
                core.file_hash(output / 'frozen_predictor.pt') != core.PREDICTOR_SHA256 or
                any(p.requires_grad or p.grad is not None for p in predictor.parameters())):
                raise RuntimeError('Frozen predictor changed')
            metadata['predictor_unchanged_after_training'] = True
        metadata['training_complete_before_evaluation'] = True
        write_json(output / 'metadata.json', metadata)
        evaluations = {}
        for step in MILESTONES:
            checkpoint = output / 'checkpoints' / f'step-{step}.pt'
            milestone, saved = load_policy(checkpoint)
            if state_digest(milestone) != saved['state_sha256']:
                raise RuntimeError('Milestone reload mismatch')
            if step == 131072 and state_digest(milestone) != training['final_state_sha256']:
                raise RuntimeError('Final milestone differs from fixed final model')
            key = f'development/step-{step}'
            evaluations[key] = evaluate_split(spec, variant, milestone, output / key,
                                               deadline, 'development', step)
            write_json(output / 'evaluation_summary.json', evaluations)
        metadata['final_test_touched'] = True
        write_json(output / 'metadata.json', metadata)
        evaluations['final_test'] = evaluate_split(spec, variant, model, output / 'final_test',
                                                   deadline, 'final_test', 131072)
        write_json(output / 'evaluation_summary.json', evaluations)
        if include_rules:
            evaluations['final_rules'] = evaluate_split(spec, 'rules', None, output / 'final_rules',
                                                        deadline, 'final_test', 131072)
        write_json(output / 'evaluation_summary.json', evaluations)
        totals = {key: training[key] + sum(item['compute'][key] for item in evaluations.values())
                  for key in ('live_control_steps', 'live_physics_steps', 'label_control_steps', 'label_physics_steps')}
        write_json(output / 'metrics.json', {'experiment_id': spec['experiment_id'], 'variant': variant,
            'training_seed': seed, 'evidence_label': 'PRELIMINARY_NON_PAPER', 'training': training,
            'evaluations': evaluations, 'compute_totals': totals,
            'dense_teacher_acquisition': 'Separate immutable E0003 acquisition; not included in these totals',
            'final_test_touched': True, 'checkpoint_selection': 'fixed_final_step_only'})
        metadata.update(status='complete', checkpoint_sha256=core.file_hash(output / 'model.pt'))
    except Exception:
        metadata.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic() - started
        write_json(output / 'metadata.json', metadata)
        hashes = {str(path.relative_to(output)): core.file_hash(path) for path in sorted(output.rglob('*'))
                  if path.is_file() and path.name != 'artifact_hashes.json'}
        write_json(output / 'artifact_hashes.json', hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', choices=('none', 'analytic', 'dense', 'sparse'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-rules', action='store_true')
    args = parser.parse_args()
    run(args.config, args.variant, args.seed, args.output, args.include_rules)


if __name__ == '__main__':
    main()
