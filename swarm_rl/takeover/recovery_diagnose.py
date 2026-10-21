"""E0003 fixed collection and recovery-cost prediction diagnostic.

Joint features are CURRENT actor observations only. Hidden operator metadata is
recorded for accounting, never passed to the model or geometry regressor.
Validation chooses a checkpoint and is explicitly not a final test estimate.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from .geometry import GeometryConfig, GeometryTakeoverEnv, reactive_actions
from .recovery_model import (errors, fit_ridge, geometry_features, joint_features,
                             predict, train_predictor)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_specification(specification):
    train = specification['training_scenario_seeds']
    val = specification['validation_scenario_seeds']
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError('Duplicate scenario seeds')
    if set(train) & set(val) or not train or not val:
        raise ValueError('Nonempty scenario-disjoint train and validation required')
    if any(seed < 0 or seed >= 10000 for seed in train):
        raise ValueError('Training seeds outside contract range')
    if any(seed < 10000 or seed >= 11000 for seed in val):
        raise ValueError('Validation seeds outside contract range; final test is forbidden')
    if specification['angles_degrees'] != list(range(0, 360, 45)):
        raise ValueError('E0003 requires all eight fixed angles')
    if specification['checkpoint_elapsed_seconds'] != [1.0, 3.0]:
        raise ValueError('E0003 checkpoint schedule changed')


def snapshot_digest(env):
    return hashlib.sha256(pickle.dumps(env.snapshot(), protocol=5)).hexdigest()


def label_snapshot(env):
    before = snapshot_digest(env)
    observations = env.observe().copy()
    label = env.evaluate_immediate_handback(max_seconds=10.0)
    if snapshot_digest(env) != before:
        raise RuntimeError('Label rollout modified the live simulator snapshot')
    np.testing.assert_array_equal(env.observe(), observations)
    if not label['protocol_horizon_complete'] or label['rollout_horizon'] != 10.0:
        raise RuntimeError('Incomplete recovery label horizon')
    cost = label['recovery_time'] if label['recovery_success'] else 10.0
    if not np.isfinite(cost) or not 0 <= cost <= 10.0:
        raise RuntimeError('Invalid recovery cost')
    return observations, {**label, 'cost_seconds': float(cost),
                           'snapshot_restored_exactly': True}


def collector_actions(env, observations, seed, action_rng, specification):
    if env.phase != 'takeover' or seed % 3 == 0:
        return env.reference_actions('nominal')
    actions = reactive_actions(observations, gain=specification['reactive_gain'],
                               radius=specification['reactive_range_m'],
                               residual_velocity=env.config.residual_velocity)
    if seed % 3 == 2:
        actions = np.clip(actions + action_rng.normal(
            0.0, specification['action_noise_std'], actions.shape), -1.0, 1.0)
    return actions


def collect_dataset(specification, output):
    observations, samples, episodes = [], [], []
    npz_path = output / 'dataset.npz'

    def persist():
        x = np.asarray(observations, dtype=np.float32).reshape(-1, 4, 41)
        arrays = {'observations': x, 'features': x.reshape(-1, 164),
                  'cost_seconds': np.asarray([s['cost_seconds'] for s in samples], dtype=np.float32),
                  'label_success': np.asarray([s['recovery_success'] for s in samples], dtype=bool),
                  'label_collision': np.asarray([s['collision'] for s in samples], dtype=bool),
                  'is_validation': np.asarray([s['split'] == 'validation' for s in samples], dtype=bool),
                  'scenario_seed': np.asarray([s['scenario_seed'] for s in samples], dtype=np.int64),
                  'angle_degrees': np.asarray([s['angle_degrees'] for s in samples], dtype=np.float32),
                  'checkpoint_elapsed_seconds': np.asarray([s['checkpoint_elapsed_seconds'] for s in samples], dtype=np.float32)}
        temporary = output / 'dataset.tmp.npz'
        np.savez_compressed(temporary, **arrays)
        temporary.replace(npz_path)
        write_json(output / 'samples.json', samples)
        write_json(output / 'collector_episodes.json', episodes)
        return arrays

    persist()
    for split, seed_key in [('training', 'training_scenario_seeds'),
                            ('validation', 'validation_scenario_seeds')]:
        for seed in specification[seed_key]:
            for angle_index, angle in enumerate(specification['angles_degrees']):
                config = GeometryConfig(detour_angle_degrees=angle,
                                        detour_distance_range=tuple(specification['detour_distance_range']))
                env = GeometryTakeoverEnv(config)
                obs, _ = env.reset(seed=seed)
                # This generator never shares state with operator/scenario sampling.
                noise_seed = [90173, int(seed), int(angle_index)]
                action_rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
                episode_id = f'{split}-seed-{seed}-angle-{angle}'
                row = {'episode_id': episode_id, 'split': split, 'scenario_seed': seed,
                       'angle_degrees': angle, 'collector_policy': ['nominal', 'reactive',
                       'reactive_plus_noise'][seed % 3], 'action_noise_seed': noise_seed,
                       'config': asdict(config), 'status': 'running', 'sample_indices': [],
                       'missing_snapshots': [], 'collector_physics_steps': 0,
                       'collector_control_steps': 0, 'label_physics_steps': 0}
                episodes.append(row)
                write_json(output / 'collector_episodes.json', episodes)
                scheduled = specification['checkpoint_elapsed_seconds']
                checkpoint_index = 0
                try:
                    while not (env.terminated or env.truncated) and checkpoint_index < len(scheduled):
                        if (env.phase == 'takeover' and
                                env.time - env._takeover_start + 1e-9 >= scheduled[checkpoint_index]):
                            current, label = label_snapshot(env)
                            observations.append(current)
                            index = len(samples)
                            samples.append({'index': index, 'episode_id': episode_id, 'split': split,
                                            'scenario_seed': seed, 'angle_degrees': angle,
                                            'checkpoint_elapsed_seconds': scheduled[checkpoint_index],
                                            'actual_elapsed_seconds': env.time - env._takeover_start,
                                            'source_time': env.time, 'source_physics_steps': env.physics_steps,
                                            'takeover_agent': env.takeover_agent,
                                            'observations': current.tolist(),
                                            'collector_policy': row['collector_policy'], **label})
                            row['sample_indices'].append(index)
                            row['label_physics_steps'] += label['simulated_physics_steps']
                            checkpoint_index += 1
                            if checkpoint_index == len(scheduled):
                                break
                        obs, _, _, _, _ = env.step(
                            collector_actions(env, obs, seed, action_rng, specification))
                    reason = ('early_collision' if env.collision else 'episode_truncated'
                              if env.truncated else 'takeover_ended_before_checkpoint')
                    for checkpoint in scheduled[checkpoint_index:]:
                        row['missing_snapshots'].append({'checkpoint_elapsed_seconds': checkpoint,
                                                         'reason': reason})
                    row['status'] = 'complete'
                except Exception as exc:
                    row['status'] = 'failed'
                    row['exception'] = repr(exc)
                    for checkpoint in scheduled[checkpoint_index:]:
                        row['missing_snapshots'].append({'checkpoint_elapsed_seconds': checkpoint,
                                                         'reason': 'execution_exception'})
                    raise
                finally:
                    row.update(collector_physics_steps=env.physics_steps,
                               collector_control_steps=env.control_steps, source_final_time=env.time,
                               source_collision=bool(env.collision), collision_phase=env.collision_phase,
                               collision_time=env.collision_time, final_phase=env.phase)
                    arrays = persist()
                    print(json.dumps({'episode_id': episode_id, 'status': row['status'],
                                      'samples': len(row['sample_indices']),
                                      'missing': row['missing_snapshots']}), flush=True)
    return arrays, samples, episodes


def source_metadata(specification, config_path, output):
    root = Path(__file__).resolve().parents[2]
    paths = list((root / 'swarm_rl/takeover').glob('*.py'))
    paths += list((root / 'gym_art/quadrotor_multi').rglob('*.py'))
    paths += list((root / 'tests').glob('test_*takeover*.py'))
    paths += list((root / 'tests').glob('test_recovery*.py'))
    paths.append(config_path.resolve())
    hashes = {}
    for path in sorted(set(paths)):
        if path.name.startswith('._'):
            continue
        relative = path.relative_to(root)
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for package in ['numpy', 'scipy', 'numba', 'gymnasium', 'torch', 'transforms3d']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cuda = torch.cuda.is_available()
    return {'experiment_id': 'E0003', 'status': 'running',
            'evidence_label': 'PRELIMINARY_NON_PAPER', 'specification': specification,
            'python': platform.python_version(), 'host': platform.node(), 'packages': packages,
            'cuda_available': cuda, 'torch_cuda_version': torch.version.cuda,
            'cuda_device_name': torch.cuda.get_device_name(0) if cuda else None,
            'cpu_threads': torch.get_num_threads(), 'source_sha256': hashes,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'observation_schema': 'four fixed-ID actor observations, 41 current-state values each',
            'label_reference_controller': 'nominal_zero_residual_v1',
            'ridge_protocol': 'raw four geometry features, L2 on non-intercept coefficients; prediction clipped to [0,10]',
            'limitations': ['Validation used for epoch selection; no untouched test estimate.',
                            'Single initialization and limited scenario family.',
                            'Cost is fixed-reference recovery, not learned-policy recovery.',
                            'No trained control policy or real human evaluation.',
                            'Actor observations omit attitude and motor-filter state.',
                            'Unavailable snapshots after collector crashes are retained as missing, not given fabricated labels.']}


def run(specification, config_path, output):
    started = time.monotonic()
    validate_specification(specification)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'experiment_id': 'E0003', 'status': 'initializing',
                'evidence_label': 'PRELIMINARY_NON_PAPER', 'specification': specification}
    write_json(output / 'metadata.json', metadata)
    try:
        metadata = source_metadata(specification, config_path, output)
        write_json(output / 'metadata.json', metadata)
        collection_start = time.monotonic()
        arrays, samples, episodes = collect_dataset(specification, output)
        collection_seconds = time.monotonic() - collection_start
        validation = arrays['is_validation']
        train = ~validation
        if not train.any() or not validation.any():
            raise RuntimeError('No training or validation labels remain')
        x, y = arrays['features'], arrays['cost_seconds']
        geometry = geometry_features(arrays['observations'])
        mean = float(y[train].mean())
        coefficients = fit_ridge(geometry[train], y[train], specification['ridge_l2'])
        model, training = train_predictor(x[train], y[train], x[validation], y[validation],
                                          specification, output / 'model.pt')
        predictions = {'constant': np.clip(np.full(len(y), mean), 0.0, 10.0),
                       'geometry_ridge': np.clip(geometry @ coefficients, 0.0, 10.0),
                       'neural': predict(model, x)}
        metrics = {'experiment_id': 'E0003', 'evidence_label': 'PRELIMINARY_NON_PAPER',
                   'predictors': {}, 'training': training, 'training_mean_cost': mean,
                   'ridge_coefficients': coefficients.tolist(),
                   'collection_wall_seconds': collection_seconds,
                   'scheduled_episodes': len(specification['angles_degrees']) * (
                       len(specification['training_scenario_seeds']) + len(specification['validation_scenario_seeds'])),
                   'completed_episodes': len(episodes), 'collected_snapshots': len(samples),
                   'missing_snapshots': sum(len(row['missing_snapshots']) for row in episodes),
                   'collector_physics_steps': sum(row['collector_physics_steps'] for row in episodes),
                   'collector_control_steps': sum(row['collector_control_steps'] for row in episodes),
                   'label_physics_steps': sum(row['label_physics_steps'] for row in episodes),
                   'train_label_successes': int(arrays['label_success'][train].sum()),
                   'validation_label_successes': int(arrays['label_success'][validation].sum()),
                   'validation_checkpoint_selected': True, 'final_test_touched': False}
        for name, prediction in predictions.items():
            metrics['predictors'][name] = {
                'training': errors(y[train], prediction[train]),
                'validation': errors(y[validation], prediction[validation]),
                'validation_label_success': errors(y[validation & arrays['label_success']],
                                                    prediction[validation & arrays['label_success']]),
                'validation_label_failure': errors(y[validation & ~arrays['label_success']],
                                                    prediction[validation & ~arrays['label_success']])}
        np.savez_compressed(output / 'predictions.npz', **predictions)
        write_json(output / 'metrics.json', metrics)
        metadata.update(status='complete', wall_seconds=time.monotonic() - started,
                        checkpoint_sha256=hashlib.sha256((output / 'model.pt').read_bytes()).hexdigest(),
                        dataset_sha256=hashlib.sha256((output / 'dataset.npz').read_bytes()).hexdigest(),
                        collection_wall_seconds=collection_seconds,
                        training_wall_seconds=training['training_wall_seconds'])
        write_json(output / 'metadata.json', metadata)
        print(json.dumps({'status': 'complete', 'metrics': metrics['predictors'],
                          'wall_seconds': metadata['wall_seconds']}), flush=True)
    except Exception:
        metadata.update(status='failed', wall_seconds=time.monotonic() - started,
                        traceback=traceback.format_exc())
        write_json(output / 'metadata.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.config, args.output)


if __name__ == '__main__':
    main()
