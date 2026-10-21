"""Prespecified paired delay sweep with source snapshots and raw trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from .core import DelayedChannel, POLICIES, RouteEnv


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def line(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(output):
    repo = Path(__file__).resolve().parents[2]
    git = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    (output / 'source').mkdir()
    sources = list(Path(__file__).parent.glob('*.py')) + [repo / 'swarm_rl/takeover/env.py',
        repo / 'swarm_rl/takeover/policy.py']
    for source in sources:
        shutil.copyfile(source, output / 'source' / source.name)
    return {'source_commit': git('rev-parse', 'HEAD'), 'source_status': git('status', '--porcelain'),
        'source_sha256': {str(p.relative_to(repo)): fingerprint(p) for p in sources}}


def run_case(case, output, deadline, learned=None):
    env = RouteEnv(case['geometry'])
    env.reset(seed=case['scenario_seed'])
    channel = DelayedChannel(case['delay_steps'])
    squared = {'raw': 0.0, 'cv': 0.0}
    estimates = 0
    ticks = 0
    intervention_integral = 0.0
    case_id = f"{case['policy']}-{case['geometry']}-d{case['delay_steps']}-s{case['scenario_seed']}"
    if learned is not None:
        from .learning import History
        history = History(learned.frames)
    with gzip.open(Path(output) / 'trajectories' / (case_id + '.jsonl.gz'), 'wt', compresslevel=1) as stream:
        while not (env.terminated or env.truncated):
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline; preserve partial records')
            raw, raw_error = channel.observation(env, 'raw')
            cv, cv_error = channel.observation(env, 'cv')
            for name, error in (('raw', raw_error), ('cv', cv_error)):
                squared[name] += error['peer_position_squared_error_sum']
            estimates += raw_error['peer_estimates']
            if learned is None:
                estimator, controller = POLICIES[case['policy']]
                obs = raw if estimator == 'raw' else cv if estimator == 'cv' else channel.observation(env, 'oracle')[0]
                action = np.zeros((4, 3)) if controller is None else controller(obs, env.config)
            else:
                import torch
                features = history.push(raw)
                with torch.no_grad():
                    action = learned.distribution(torch.from_numpy(features)).mean.tanh().numpy()
            before = env.time
            line(stream, {'pre_time': before, 'positions': env.positions.tolist(),
                'velocities': env.velocities.tolist(), 'delayed_observations': raw.tolist(),
                'packet_age': raw_error['packet_age'], 'actions': action.tolist()})
            _, _, _, _, metrics = env.step(action)
            ticks += 1
            intervention_integral += float(np.linalg.norm(action, axis=1).mean() * (env.time - before))
        line(stream, {'terminal': True, 'time': env.time, 'positions': env.positions.tolist(),
            'velocities': env.velocities.tolist(), 'metrics': metrics})
    result = {**case, **metrics, 'status': 'complete', 'split': 'development_diagnostic',
        'raw_peer_position_rmse': float(np.sqrt(squared['raw'] / estimates)),
        'cv_peer_position_rmse': float(np.sqrt(squared['cv'] / estimates)),
        'perception_evaluation': 'raw and cv paired on this controller own closed-loop states',
        'correction_integral': intervention_integral, 'control_steps': ticks,
        'initial_state': {'goals': env.goals.tolist(), 'cruise_speeds': env.cruise_speeds.tolist(),
                          'launch_times': env.launch_times.tolist()}}
    write(Path(output) / 'cases' / (case_id + '.json'), result)
    return result


def aggregate(cases):
    rows = []
    for key in sorted({(c['geometry'], c['policy'], c['delay_steps']) for c in cases}):
        group = [c for c in cases if (c['geometry'], c['policy'], c['delay_steps']) == key]
        rows.append({'geometry': key[0], 'policy': key[1], 'delay_steps': key[2], 'n': len(group),
            'successes': sum(c['success'] for c in group), 'collisions': sum(c['collision'] for c in group),
            'mean_failure_inclusive_completion': float(np.mean([c['failure_inclusive_completion_cost'] for c in group])),
            'raw_position_rmse_mean': float(np.mean([c['raw_peer_position_rmse'] for c in group])),
            'cv_position_rmse_mean': float(np.mean([c['cv_peer_position_rmse'] for c in group]))})
    return rows


def gates(cases, spec):
    """Engineering screening thresholds, NOT tests of statistical significance."""
    index = {(r['geometry'], r['policy'], r['delay_steps']): r for r in aggregate(cases)}
    results = []
    for delay in spec['delay_steps']:
        if delay == 0 or delay in spec['stress_only_delay_steps']:
            continue
        raw = index[('crossing', 'reactive_raw', delay)]
        zero = index[('crossing', 'reactive_raw', 0)]
        oracle = index[('crossing', 'reactive_oracle', delay)]
        cv = index[('crossing', 'predictive_cv', delay)]
        predictive_oracle = index[('crossing', 'predictive_oracle', delay)]
        deployable = [index[('crossing', p, delay)] for p in ('reactive_raw', 'reactive_cv', 'predictive_cv')]
        loss = raw['collisions'] - zero['collisions'] >= 2 or zero['successes'] - raw['successes'] >= 2
        benefit = oracle['collisions'] <= raw['collisions'] - 2 or oracle['successes'] >= raw['successes'] + 2
        prediction_gap = (predictive_oracle['collisions'] <= cv['collisions'] - 2 or
                          predictive_oracle['successes'] >= cv['successes'] + 2)
        # Requiring better safety and no worse completion avoids rewarding stalls.
        beyond_simple = all(predictive_oracle['collisions'] <= r['collisions'] and
            predictive_oracle['successes'] >= r['successes'] + 2 for r in deployable)
        results.append({'delay_steps': delay, 'delay_difficulty': bool(loss),
            'same_reactive_controller_information_benefit': bool(benefit),
            'same_predictive_controller_information_benefit': bool(prediction_gap),
            'oracle_predictive_exceeds_all_simple_baselines': bool(beyond_simple)})
    return {'threshold': 'At least two cases out of eight; exploratory gate, not statistical significance.',
        'per_delay': results, 'oracle_warning': 'Same-controller fresh-state reference, not an optimal control upper bound.',
        'learning_prediction_candidate': any(r['delay_difficulty'] and
            r['same_predictive_controller_information_benefit'] and
            r['oracle_predictive_exceeds_all_simple_baselines'] for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/delay_awareness_pilot.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if args.smoke:
        spec.update(scenario_seeds=[71999], delay_steps=[0, 2], workers=2)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    (output / 'trajectories').mkdir()
    metadata = {**source_record(output), 'status': 'running', 'smoke': args.smoke,
        'numpy': np.__version__, 'kind': 'new_route_task_no_operator_or_takeover'}
    write(output / 'metadata.json', metadata)
    write(output / 'effective_config.json', spec)
    started = time.monotonic()
    cases = []
    try:
        if metadata['source_status'] and not args.smoke:
            raise RuntimeError('Commit source before diagnostic')
        jobs = [{'policy': p, 'geometry': g, 'delay_steps': d, 'scenario_seed': s}
            for p in spec['policies'] for g in spec['geometries']
            for d in spec['delay_steps'] for s in spec['scenario_seeds']]
        deadline = started + spec['max_wall_seconds']
        with ProcessPoolExecutor(max_workers=spec['workers']) as pool:
            futures = [pool.submit(run_case, case, output, deadline) for case in jobs]
            for future in as_completed(futures):
                cases.append(future.result())
                write(output / 'progress.json', {'completed': len(cases), 'total': len(jobs),
                    'wall_seconds': time.monotonic() - started})
        cases.sort(key=lambda c: (c['geometry'], c['policy'], c['delay_steps'], c['scenario_seed']))
        write(output / 'all_cases.json', cases)
        write(output / 'summary.json', aggregate(cases))
        if not args.smoke:
            write(output / 'gates.json', gates(cases, spec))
        metadata.update(status='complete', cases=len(cases))
    except Exception:
        metadata.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic() - started
        write(output / 'metadata.json', metadata)
    write(output / 'artifact_hashes.json', {str(p.relative_to(output)): fingerprint(p)
        for p in output.rglob('*') if p.is_file()})


if __name__ == '__main__':
    main()
