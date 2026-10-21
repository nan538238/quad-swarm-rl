"""Predeclared E0002 geometry sweep. No training or method selection."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from .geometry import GeometryConfig, GeometryTakeoverEnv, reactive_actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    specification = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / 'swarm_rl/takeover').glob('*.py'))
    sources += sorted((root / 'gym_art/quadrotor_multi').rglob('*.py'))
    sources += sorted((root / 'tests').glob('test_takeover*.py'))
    sources += [args.config.resolve()]
    hashes = {}
    for source in sources:
        if source.name.startswith('._'):
            continue
        relative = source.relative_to(root)
        copy = args.output / 'source' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        'evidence_label': 'PRELIMINARY_NON_PAPER', 'status': 'running',
        'specification': specification, 'python': platform.python_version(),
        'host': platform.node(), 'source_sha256': hashes,
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'numba', 'gymnasium']},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'limitations': ['Engineering seeds, no independent training or held-out testing.',
                        'Zero noise, independent dynamics with contact termination.',
                        'Geometric development sweep does not establish method novelty.',
                        'Progress lag is observed duration only; early crashes cannot earn a good lag score.'],
    }
    meta_path = args.output / 'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2))
    rows = []
    metrics_path = args.output / 'metrics.json'
    metrics_path.write_text('[]')
    for angle in specification['angles_degrees']:
        for seed in specification['scenario_seeds']:
            for mode in specification['modes']:
                config = GeometryConfig(detour_angle_degrees=angle,
                                        detour_distance_range=tuple(specification['detour_distance_range']))
                env = GeometryTakeoverEnv(config)
                obs, _ = env.reset(seed=seed)
                name = f'angle-{angle}-seed-{seed}-{mode}'
                history, lag_integral, team_return = [], 0.0, 0.0
                previous_time = env.time
                previous_lag = float(np.maximum(env.slot_targets[:, 0] - env.positions[:, 0], 0).mean())
                while True:
                    history.append({'time': env.time, 'phase': env.phase,
                                    'positions': env.positions.tolist(), 'velocities': env.velocities.tolist(),
                                    'slot_targets': env.slot_targets.tolist(),
                                    'takeover_flags': env.takeover_flags.tolist()})
                    if mode in ('nominal', 'wait'):
                        actions = env.reference_actions(mode)
                    elif mode == 'reactive_all' or env.phase == 'handback':
                        actions = reactive_actions(obs, gain=specification['reactive_gain'],
                                                   radius=specification['reactive_range_m'],
                                                   residual_velocity=config.residual_velocity)
                    else:
                        actions = env.reference_actions('nominal')
                    obs, reward, done, truncated, _ = env.step(actions)
                    team_return += reward
                    lag = float(np.maximum(env.slot_targets[:, 0] - env.positions[:, 0], 0).mean())
                    lag_integral += (previous_lag + lag) * 0.5 * (env.time - previous_time)
                    previous_time, previous_lag = env.time, lag
                    if done or truncated:
                        history.append({'time': env.time, 'phase': env.phase,
                                        'positions': env.positions.tolist(), 'velocities': env.velocities.tolist(),
                                        'slot_targets': env.slot_targets.tolist(),
                                        'takeover_flags': env.takeover_flags.tolist()})
                        break
                with gzip.open(args.output / f'{name}.json.gz', 'wt') as f:
                    json.dump(history, f)
                metric = env.metrics()
                complete = bool(metric['recovery_success'] and metric['check_completed'] and not metric['collision'])
                row = {'name': name, 'angle_degrees': angle, 'seed': seed, 'mode': mode,
                       'takeover_agent': env.takeover_agent, **metric,
                       'collision_phase': env.collision_phase, 'collision_time': env.collision_time,
                       'scenario_success': complete,
                       'recovery_time_with_failure_penalty': metric['recovery_time'] if complete else config.recovery_timeout,
                       'positive_agent_patrol_lag_integral_m_s': lag_integral,
                       'lag_observation_seconds': env.time, 'return': team_return,
                       'config': asdict(config)}
                rows.append(row)
                metrics_path.write_text(json.dumps(rows, indent=2))
                print(json.dumps({k: row[k] for k in ['name', 'collision_phase', 'scenario_success',
                                                      'recovery_time_with_failure_penalty']}), flush=True)
    expected = len(specification['angles_degrees']) * len(specification['scenario_seeds']) * len(specification['modes'])
    assert len(rows) == expected
    metadata.update(status='complete', wall_seconds=time.monotonic() - started,
                    episode_count=len(rows), physics_steps=sum(x['physics_steps'] for x in rows))
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'status': 'complete', 'episodes': len(rows), 'wall_seconds': metadata['wall_seconds']}))


if __name__ == '__main__':
    main()
