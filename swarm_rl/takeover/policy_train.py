"""Bounded CP-C2-01 training and fixed-checkpoint validation; preliminary only.

All 32768 live control transitions count, including nominal prelude and handback.
The actor is updated only for autonomous rows in pre-step takeover. Rewards and
the separate team critic include the complete episode through 24s or collision.
"""
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
import traceback

import numpy as np
import torch

from .geometry import GeometryConfig, GeometryTakeoverEnv, reactive_actions
from .policy import TeamPolicy, load_policy, state_digest, update_policy
from .recovery_diagnose import label_snapshot, write_json
from .recovery_model import joint_features, load_predictor, predict


PREDICTOR_SHA256 = '3eceb9217ee4c7b557b738213e6e33c54a2d5a9ed15047236d1151fdb07c52d0'
RULES = ('nominal', 'wait_takeover', 'reactive_takeover', 'reactive_all')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')


def validate_specification(specification):
    fixed = {'protocol_id': 'CP-C2-01', 'config_id': 'takeover-policy-cp-c2-01-v1',
             'angles_degrees': list(range(0, 360, 45)), 'training_seed_range': [300, 999],
             'validation_scenario_seeds': list(range(10020, 10030)),
             'detour_distance_range': [1.0, 2.6], 'episode_seconds': 24.0,
             'torch_seed': 0, 'learning_rate': 0.0003, 'gamma': 0.995,
             'gae_lambda': 0.95, 'clip': 0.2, 'epochs': 4, 'minibatch_timesteps': 256,
             'rollout_timesteps': 2048, 'entropy_coefficient': 0.001,
             'value_coefficient': 0.5, 'max_grad_norm': 0.5, 'training_steps': 32768,
             'device': 'cpu', 'torch_threads': 1, 'auxiliary_duration_scale_seconds': 4.5,
             'reactive_gain': 1.5, 'reactive_range_m': 0.9,
             'training_schedule': 'seed=300+(episode_index//8)%700; angle=45*(episode_index%8)'}
    for key, expected in fixed.items():
        if specification.get(key) != expected:
            raise ValueError(f'CP-C2-01 fixed field changed: {key}')
    if specification['experiment_id'] not in ('E0004', 'E0005'):
        raise ValueError('Unknown experiment identity')
    if not 0 < specification['max_wall_seconds'] <= 2700:
        raise ValueError('Invalid bounded wall cap')


def training_scenario(index):
    if index < 0:
        raise ValueError('Negative episode index')
    return 300 + (index // 8) % 700, 45 * (index % 8)


def make_env(specification, angle):
    return GeometryTakeoverEnv(GeometryConfig(
        detour_angle_degrees=angle,
        detour_distance_range=tuple(specification['detour_distance_range']),
        episode_seconds=specification['episode_seconds']))


def failure_inclusive_recovery_cost(metrics):
    if metrics['collision'] or not metrics['recovery_success']:
        return 10.0
    return float(metrics['recovery_time'])


def terminal_mission_reward(metrics, done):
    return (2.0 * float(metrics['check_completed']) - failure_inclusive_recovery_cost(metrics)
            if done else 0.0)


def analytic_cost(observations):
    slot = np.linalg.norm(observations[:, 3:6], axis=-1).max()
    velocity = np.linalg.norm(observations[:, :3] - observations[:, 9:12], axis=-1).max()
    return float(np.clip(slot / 0.6 + velocity / 0.8, 0, 10))


def auxiliary_reward(variant, pre_phase, observations, env, predictor, specification):
    """Post-step signal; all scheduling is based on the pre-step phase."""
    if pre_phase != 'takeover' or variant == 'none':
        return 0.0, None
    if variant == 'sparse':
        if env.phase != 'handback' or env.terminated or env.truncated or env.collision:
            return 0.0, None
        started = time.monotonic()
        _, label = label_snapshot(env)
        # Prioritize contact even if recovery was reached earlier within the
        # same final control interval. Preserve the E0003 primitive unchanged.
        if label['collision']:
            label['cost_seconds'] = 10.0
        # Existing reference primitive stops when recovery is decided. Every
        # completed control transition contains ceil(physics/5) control calls.
        ratio = round(env.config.control_dt / env.config.physics_dt)
        label['simulated_control_steps'] = int(np.ceil(label['simulated_physics_steps'] / ratio))
        label['wall_seconds'] = time.monotonic() - started
        return -label['cost_seconds'], label
    if env.collision:
        cost = 10.0
    elif variant == 'analytic':
        cost = analytic_cost(observations)
    elif variant == 'dense':
        cost = float(predict(predictor, joint_features(observations)[None])[0])
    else:
        raise ValueError(f'Unknown reward variant {variant}')
    return -env.config.control_dt / specification['auxiliary_duration_scale_seconds'] * cost, None


def lag(env):
    return float(np.maximum(env.slot_targets[:, 0] - env.positions[:, 0], 0).mean())


def episode_metrics(env, integrated_lag):
    values = env.metrics()
    return {**values, 'scenario_success': bool(values['check_completed'] and
            values['recovery_success'] and not values['collision']),
            'failure_inclusive_recovery_cost': failure_inclusive_recovery_cost(values),
            'collision_phase': env.collision_phase, 'collision_time': env.collision_time,
            'integrated_patrol_lag': integrated_lag,
            'positive_agent_patrol_lag_integral_m_s': integrated_lag,
            'lag_observation_seconds': env.time}


def reference_action(policy, env, observations, specification):
    if policy == 'nominal':
        return env.reference_actions('nominal')
    if policy == 'wait_takeover':
        actions = env.reference_actions('wait')
    elif policy in ('reactive_takeover', 'reactive_all'):
        if policy == 'reactive_takeover' and env.phase != 'takeover':
            return env.reference_actions('nominal')
        actions = reactive_actions(observations, gain=specification['reactive_gain'],
                                   radius=specification['reactive_range_m'],
                                   residual_velocity=env.config.residual_velocity)
    else:
        raise ValueError(policy)
    actions[env.takeover_flags.astype(bool)] = 0.0
    return actions


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Prespecified wall cap reached; partial artifacts retained')


def train(specification, variant, output, predictor, deadline, checkpoint_callback=None):
    torch.manual_seed(specification['torch_seed'])
    model = TeamPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=specification['learning_rate'])
    initial_digest = state_digest(model)
    initial_parameters = {key: value.detach().clone() for key, value in model.named_parameters()}
    # Separate deterministic RNGs: model init, action sampling, entropy samples,
    # minibatch shuffling and simulator episode sampling never share mutable state.
    action_rng = torch.Generator().manual_seed(specification['torch_seed'])
    entropy_rng = torch.Generator().manual_seed(1000 + specification['torch_seed'])
    shuffle_rng = np.random.default_rng(specification['torch_seed'])
    counts = {'live_control_steps': 0, 'live_physics_steps': 0, 'label_control_steps': 0,
              'label_physics_steps': 0, 'label_wall_seconds': 0.0,
              'active_actor_rows': 0, 'active_takeover_timesteps': 0,
              'completed_episodes': 0, 'updates': 0}
    write_json(output / 'initialization.json', {'state_sha256': initial_digest,
               'initializer': f"torch.nn.Linear defaults; seed{specification['torch_seed']}; logstd=-1",
               'rng_seeds': {'initial_weights': specification['torch_seed'],
                             'action': specification['torch_seed'], 'shuffle': specification['torch_seed'],
                             'entropy': 1000 + specification['torch_seed']},
               'schedule': specification['training_schedule'],
               'advantage_normalization': 'over active actor rows per rollout, population std',
               'entropy': 'reparameterized MC differential entropy of current tanh Gaussian'})
    episode_index = 0
    seed, angle = training_scenario(episode_index)
    env = make_env(specification, angle)
    observations, _ = env.reset(seed=seed)
    integrated_lag, previous_lag = 0.0, lag(env)
    episode_rewards = {'environment': 0.0, 'terminal_mission': 0.0, 'auxiliary': 0.0, 'total': 0.0}
    episode_history = []
    started = time.monotonic()
    with (gzip.open(output / 'training_transitions.jsonl.gz', 'wt', compresslevel=1) as transitions,
          (output / 'training_episodes.jsonl').open('w') as episodes,
          (output / 'training_updates.jsonl').open('w') as updates,
          (output / 'training_labels.jsonl').open('w') as labels):
        try:
            while counts['live_control_steps'] < specification['training_steps']:
                rollout = {key: [] for key in ('observations', 'latents', 'log_probs',
                           'masks', 'rewards', 'values', 'episode_ends')}
                length = min(specification['rollout_timesteps'],
                             specification['training_steps'] - counts['live_control_steps'])
                for _ in range(length):
                    check_deadline(deadline)
                    tensor = torch.as_tensor(observations)
                    with torch.no_grad():
                        actions, latent, log_probs, mask, value = model.sample(tensor, action_rng)
                    pre_phase, pre_time, pre_physics = env.phase, env.time, env.physics_steps
                    next_observations, env_reward, terminal, truncated, info = env.step(actions.numpy())
                    done = terminal or truncated
                    terminal_reward = terminal_mission_reward(info, done)
                    auxiliary, label = auxiliary_reward(variant, pre_phase, next_observations,
                                                        env, predictor, specification)
                    reward = env_reward + terminal_reward + auxiliary
                    now_lag = lag(env)
                    integrated_lag += (previous_lag + now_lag) * 0.5 * (env.time - pre_time)
                    previous_lag = now_lag
                    counts['live_control_steps'] += 1
                    counts['live_physics_steps'] += env.physics_steps - pre_physics
                    counts['active_actor_rows'] += int(mask.sum())
                    counts['active_takeover_timesteps'] += int(mask.any())
                    if label is not None:
                        counts['label_control_steps'] += label['simulated_control_steps']
                        counts['label_physics_steps'] += label['simulated_physics_steps']
                        counts['label_wall_seconds'] += label['wall_seconds']
                        json_line(labels, {'episode_index': episode_index, 'scenario_seed': seed,
                                          'angle_degrees': angle, 'source_time': env.time, **label})
                        labels.flush()
                    for key, component in [('environment', env_reward), ('terminal_mission', terminal_reward),
                                           ('auxiliary', auxiliary), ('total', reward)]:
                        episode_rewards[key] += component
                    row = {'step': counts['live_control_steps'], 'episode_index': episode_index,
                           'scenario_seed': seed, 'angle_degrees': angle, 'pre_phase': pre_phase,
                           'pre_time': pre_time, 'time': env.time, 'phase': env.phase,
                           'observations': observations.tolist(), 'actions': actions.tolist(),
                           'latents': latent.tolist(), 'log_probs': log_probs.tolist(),
                           'mask': mask.tolist(), 'value': float(value),
                           'environment_reward': env_reward, 'terminal_mission_reward': terminal_reward,
                           'auxiliary_reward': auxiliary, 'reward': reward,
                           'terminated': terminal, 'truncated': truncated,
                           'positions': env.positions.tolist(), 'velocities': env.velocities.tolist(),
                           'integrated_patrol_lag': integrated_lag}
                    json_line(transitions, row)
                    episode_history.append(row)
                    for key, data in [('observations', observations.copy()), ('latents', latent.numpy()),
                                      ('log_probs', log_probs.numpy()), ('masks', mask.numpy()),
                                      ('rewards', reward), ('values', float(value)), ('episode_ends', done)]:
                        rollout[key].append(data)
                    observations = next_observations
                    if done:
                        json_line(episodes, {'episode_index': episode_index, 'scenario_seed': seed,
                                  'angle_degrees': angle, 'policy': variant,
                                  'status': 'complete', 'reward_components': episode_rewards,
                                  **episode_metrics(env, integrated_lag)})
                        episodes.flush()
                        counts['completed_episodes'] += 1
                        episode_index += 1
                        # Do not fabricate an additional unfinished episode at an
                        # exact terminal training boundary.
                        if counts['live_control_steps'] < specification['training_steps']:
                            seed, angle = training_scenario(episode_index)
                            env = make_env(specification, angle)
                            observations, _ = env.reset(seed=seed)
                            integrated_lag, previous_lag = 0.0, lag(env)
                            episode_rewards = {key: 0.0 for key in episode_rewards}
                            episode_history = []
                with torch.no_grad():
                    rollout['bootstrap'] = (0.0 if rollout['episode_ends'][-1] else
                                            float(model.value(torch.as_tensor(observations))))
                check_deadline(deadline)
                report = update_policy(model, optimizer, rollout, specification, shuffle_rng, entropy_rng)
                counts['updates'] += 1
                if checkpoint_callback is not None:
                    checkpoint_callback(model, counts)
                json_line(updates, {'update': counts['updates'], **counts,
                                   'wall_seconds': time.monotonic() - started, **report})
                updates.flush()
                transitions.flush()
                write_json(output / 'training_progress.json', counts)
                print(json.dumps({'variant': variant, 'update': counts['updates'],
                                  'steps': counts['live_control_steps'],
                                  'wall_seconds': time.monotonic() - started}), flush=True)
        finally:
            write_json(output / 'training_progress.json', counts)
            incomplete = not (env.terminated or env.truncated)
            write_json(output / 'training_incomplete_episode.json', {
                'present': incomplete, 'episode_index': episode_index,
                'scenario_seed': seed, 'angle_degrees': angle,
                'status': 'unfinished' if incomplete else 'last_episode_complete',
                'observations': observations.tolist() if incomplete else None,
                'reward_components': episode_rewards, **episode_metrics(env, integrated_lag)})
            with gzip.open(output / 'training_incomplete_trajectory.jsonl.gz', 'wt', compresslevel=1) as stream:
                if incomplete:
                    for row in episode_history:
                        json_line(stream, row)
    final_digest = state_digest(model)
    changes = {group: float(torch.sqrt(sum((value.detach() - initial_parameters[key]).square().sum()
               for key, value in model.named_parameters() if (key.startswith('critic') == (group == 'critic')))))
               for group in ('actor', 'critic')}
    if counts['live_control_steps'] != specification['training_steps']:
        raise RuntimeError('Fixed live-step count not reached')
    if min(changes.values()) <= 0:
        raise RuntimeError('Actor or critic did not update')
    torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'specification': specification, 'variant': variant,
                'checkpoint_selection': 'fixed_final_step_only', 'counts': counts,
                'initial_state_sha256': initial_digest, 'final_state_sha256': final_digest,
                'action_rng_state': action_rng.get_state(), 'entropy_rng_state': entropy_rng.get_state(),
                'shuffle_rng_state': shuffle_rng.bit_generator.state}, output / 'model.pt')
    reloaded, _ = load_policy(output / 'model.pt')
    with torch.no_grad():
        original_actions = model.deterministic(torch.as_tensor(observations))
        loaded_actions = reloaded.deterministic(torch.as_tensor(observations))
    torch.testing.assert_close(original_actions, loaded_actions, rtol=0, atol=0)
    if state_digest(reloaded) != final_digest:
        raise RuntimeError('Checkpoint reload changed parameters')
    return reloaded, {**counts, 'wall_seconds': time.monotonic() - started,
                      'initial_state_sha256': initial_digest, 'final_state_sha256': final_digest,
                      'parameter_change_l2': changes, 'checkpoint_reload_max_absolute_error': 0.0,
                      'incomplete_episode_present': not (env.terminated or env.truncated)}


def evaluate(specification, variant, model, output, deadline, *, split='validation', scenario_seeds=None):
    cases = []
    policies = RULES if variant == 'rules' else (variant,)
    counts = {'live_control_steps': 0, 'live_physics_steps': 0,
              'label_control_steps': 0, 'label_physics_steps': 0}
    started = time.monotonic()
    with gzip.open(output / 'validation_trajectories.jsonl.gz', 'wt', compresslevel=1) as trajectories:
        for policy in policies:
            for seed in (specification['validation_scenario_seeds'] if scenario_seeds is None else scenario_seeds):
                for angle in specification['angles_degrees']:
                    check_deadline(deadline)
                    env = make_env(specification, angle)
                    observations, _ = env.reset(seed=seed)
                    episode_id = f'{policy}-seed-{seed}-angle-{angle}'
                    case = {'episode_id': episode_id, 'policy': policy, 'scenario_seed': seed,
                            'angle_degrees': angle, 'split': split, 'status': 'running',
                            'config': asdict(env.config), 'deterministic': True}
                    cases.append(case)
                    integrated_lag, previous_lag, total_reward = 0.0, lag(env), 0.0
                    json_line(trajectories, {'episode_id': episode_id, 'time': env.time,
                              'phase': env.phase, 'positions': env.positions.tolist(),
                              'velocities': env.velocities.tolist(), 'actions': None})
                    try:
                        while not (env.terminated or env.truncated):
                            check_deadline(deadline)
                            if variant == 'rules':
                                actions = reference_action(policy, env, observations, specification)
                            else:
                                with torch.no_grad():
                                    actions = model.deterministic(torch.as_tensor(observations)).numpy()
                            pre_time, pre_physics, pre_phase = env.time, env.physics_steps, env.phase
                            observations, reward, terminal, truncated, info = env.step(actions)
                            total_reward += reward + terminal_mission_reward(info, terminal or truncated)
                            now_lag = lag(env)
                            integrated_lag += (previous_lag + now_lag) * 0.5 * (env.time - pre_time)
                            previous_lag = now_lag
                            counts['live_control_steps'] += 1
                            counts['live_physics_steps'] += env.physics_steps - pre_physics
                            json_line(trajectories, {'episode_id': episode_id, 'time': env.time,
                                      'phase': env.phase, 'pre_phase': pre_phase,
                                      'positions': env.positions.tolist(), 'velocities': env.velocities.tolist(),
                                      'actions': actions.tolist(), 'override_mask': env.last_override_mask.tolist(),
                                      'environment_reward': reward, 'terminated': terminal, 'truncated': truncated})
                        case.update(status='complete')
                    except Exception as exc:
                        case.update(status='failed', exception=repr(exc))
                        raise
                    finally:
                        case.update(**episode_metrics(env, integrated_lag), common_return=total_reward)
                        write_json(output / 'validation_cases.json', cases)
                        write_json(output / 'validation_progress.json', counts)
                        trajectories.flush()
                    print(json.dumps({'validation_case': episode_id, 'status': case['status']}), flush=True)
    return cases, {**counts, 'wall_seconds': time.monotonic() - started}


def summarize(cases):
    summary = {}
    for name in sorted({case['policy'] for case in cases}):
        rows = [case for case in cases if case['policy'] == name]
        summary[name] = {'count': len(rows), 'successes': sum(row['scenario_success'] for row in rows),
                         'collisions': sum(row['collision'] for row in rows),
                         'mean_failure_inclusive_recovery_cost': float(np.mean([
                             row['failure_inclusive_recovery_cost'] for row in rows])),
                         'per_angle': {str(angle): {'count': sum(r['angle_degrees'] == angle for r in rows),
                             'successes': sum(r['scenario_success'] for r in rows if r['angle_degrees'] == angle),
                             'collisions': sum(r['collision'] for r in rows if r['angle_degrees'] == angle),
                             'mean_failure_inclusive_recovery_cost': float(np.mean([
                                 r['failure_inclusive_recovery_cost'] for r in rows if r['angle_degrees'] == angle]))}
                             for angle in range(0, 360, 45)}}
    return summary


def source_metadata(specification, config_path, output, variant):
    root = Path(__file__).resolve().parents[2]
    paths = list((root / 'swarm_rl/takeover').glob('*.py'))
    paths += list((root / 'gym_art').rglob('*.py'))
    paths += list((root / 'tests').glob('test_*.py'))
    paths += [config_path.resolve(), root / 'swarm_rl/takeover/README.md']
    hashes = {}
    for path in sorted(set(paths)):
        if path.name.startswith('._'):
            continue
        relative = path.relative_to(root)
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(relative)] = file_hash(path)
    packages = {}
    for package in ('numpy', 'scipy', 'numba', 'gymnasium', 'torch', 'transforms3d'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'experiment_id': specification['experiment_id'], 'variant': variant,
            'protocol_id': specification['protocol_id'], 'config_id': specification['config_id'],
            'evidence_label': 'PRELIMINARY_NON_PAPER', 'status': 'running',
            'specification': specification, 'source_sha256': hashes,
            'config_sha256': file_hash(config_path), 'python': platform.python_version(),
            'host': platform.node(), 'packages': packages, 'device': 'cpu',
            'torch_threads': torch.get_num_threads(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
            'final_test_touched': False, 'checkpoint_selection': 'fixed_final_step_only',
            'cost_definition': '10 if any collision or no recovery, otherwise recovery seconds; check failure separate',
            'lag_definition': 'control-boundary trapezoid integral of mean positive per-agent x slot lag; observed duration',
            'observation_schema': '4x41 current actor observations only; no info or future fields',
            'limitations': ['Single seed and fixed short optimization budget; no convergence claim.',
                           'Validation only; no final test or paper evidence.',
                           'Auxiliary timing/magnitude differ and may confound information effect.',
                           'Sparse labels stop at first nominal recovery decision.',
                           'Patrol lag comparisons require paired common-success cases.']}


def run(specification, config_path, variant, output):
    started = time.monotonic()
    validate_specification(specification)
    if variant == 'sparse' and specification['experiment_id'] != 'E0005':
        raise ValueError('Sparse requires separately activated E0005 configuration')
    if variant != 'sparse' and specification['experiment_id'] != 'E0004':
        raise ValueError('E0005 reuses E0004 immutable baselines; no duplicate baseline execution')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    deadline = started + specification['max_wall_seconds']
    metadata = {'status': 'initializing', 'variant': variant,
                'experiment_id': specification['experiment_id']}
    write_json(output / 'metadata.json', metadata)
    try:
        metadata = source_metadata(specification, config_path, output, variant)
        predictor = None
        predictor_path = None
        predictor_state_digest = None
        if variant == 'dense':
            root = Path(__file__).resolve().parents[2]
            predictor_path = root / specification['dense_predictor_path']
            if file_hash(predictor_path) != PREDICTOR_SHA256:
                raise RuntimeError('Frozen E0003 checkpoint identity mismatch')
            shutil.copyfile(predictor_path, output / 'frozen_predictor.pt')
            predictor, _ = load_predictor(output / 'frozen_predictor.pt', device='cpu')
            predictor.requires_grad_(False)
            predictor_state_digest = state_digest(predictor)
            metadata.update(predictor_sha256=PREDICTOR_SHA256,
                            predictor_source=str(predictor_path),
                            predictor_state_sha256= predictor_state_digest)
        write_json(output / 'metadata.json', metadata)
        model, training = (None, None) if variant == 'rules' else train(
            specification, variant, output, predictor, deadline)
        if predictor is not None:
            if (file_hash(predictor_path) != PREDICTOR_SHA256 or
                    file_hash(output / 'frozen_predictor.pt') != PREDICTOR_SHA256 or
                    state_digest(predictor) != predictor_state_digest or
                    any(p.grad is not None or p.requires_grad for p in predictor.parameters())):
                raise RuntimeError('Frozen predictor changed or received gradients')
            metadata['predictor_unchanged_after_training'] = True
        cases, evaluation = evaluate(specification, variant, model, output, deadline)
        expected = 320 if variant == 'rules' else 80
        if len(cases) != expected or any(case['status'] != 'complete' for case in cases):
            raise RuntimeError('Incomplete fixed validation cases')
        metrics = {'experiment_id': specification['experiment_id'], 'variant': variant,
                   'evidence_label': 'PRELIMINARY_NON_PAPER', 'training': training,
                   'evaluation': evaluation, 'validation': summarize(cases),
                   'final_test_touched': False, 'checkpoint_selection': 'fixed_final_step_only'}
        write_json(output / 'metrics.json', metrics)
        metadata.update(status='complete', wall_seconds=time.monotonic() - started,
                        checkpoint_sha256=file_hash(output / 'model.pt') if model is not None else None)
        write_json(output / 'metadata.json', metadata)
        hashes = {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob('*'))
                  if path.is_file() and path.name != 'artifact_hashes.json'}
        write_json(output / 'artifact_hashes.json', hashes)
        print(json.dumps({'status': 'complete', 'variant': variant,
                          'wall_seconds': metadata['wall_seconds']}), flush=True)
    except Exception:
        metadata.update(status='failed', wall_seconds=time.monotonic() - started,
                        traceback=traceback.format_exc())
        write_json(output / 'metadata.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', choices=('none', 'analytic', 'dense', 'sparse', 'rules'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.config, args.variant, args.output)


if __name__ == '__main__':
    main()
