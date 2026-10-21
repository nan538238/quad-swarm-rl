"""E0001 paired infrastructure diagnostics; no learned-method evidence."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .env import TakeoverConfig, TakeoverPatrolEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = TakeoverConfig()
    rows = []
    for enabled, mode in [(False, 'nominal'), (True, 'nominal'), (True, 'wait')]:
        for seed in range(5):
            env = TakeoverPatrolEnv(replace(config, takeover_enabled=enabled))
            obs, info = env.reset(seed=seed)
            trajectory = []
            episode_return = 0.0
            while True:
                trajectory.append({'time': env.time, 'phase': env.phase,
                                   'positions': env.positions.tolist(),
                                   'velocities': env.velocities.tolist(),
                                   'slot_targets': env.slot_targets.tolist(),
                                   'takeover_flags': env.takeover_flags.tolist()})
                obs, reward, done, truncated, info = env.step(env.reference_actions(mode))
                episode_return += reward
                if done or truncated:
                    trajectory.append({'time': env.time, 'phase': env.phase,
                                       'positions': env.positions.tolist(),
                                       'velocities': env.velocities.tolist(),
                                       'slot_targets': env.slot_targets.tolist(),
                                       'takeover_flags': env.takeover_flags.tolist()})
                    break
            name = f'{"takeover" if enabled else "patrol"}-{mode}-{seed}'
            (args.output / f'{name}.json').write_text(json.dumps(trajectory))
            takeover_points = [p for p in trajectory if p['phase'] == 'takeover']
            row = {'name': name, 'seed': seed, 'mode': mode, 'takeover_enabled': enabled,
                   'takeover_agent': env.takeover_agent, 'return': episode_return,
                   **env.metrics()}
            if takeover_points:
                a, b = takeover_points[0], takeover_points[-1]
                row['takeover_progress'] = float(np.mean(np.asarray(b['positions'])[:, 0]) -
                                                 np.mean(np.asarray(a['positions'])[:, 0]))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2))
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'swarm_rl/takeover').glob('*.py')) + sorted((root / 'tests').glob('test_takeover*.py'))
    metadata = {'evidence_label': 'PRELIMINARY_NON_PAPER', 'config': asdict(config),
                'scenario_seeds': list(range(5)), 'wall_seconds': time.monotonic() - started,
                'python': platform.python_version(), 'host': platform.node(),
                'packages': {p: importlib.metadata.version(p) for p in
                             ['numpy', 'scipy', 'numba', 'gymnasium', 'torch']},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                'limitations': ['Independent dynamics except contact termination.',
                                'Fixed absolute time-based slots; no moving formation center.',
                                'No obstacle, downwash, motor noise, learned policies or real human.',
                                'Engineering seeds only; no held-out scientific test.']}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
