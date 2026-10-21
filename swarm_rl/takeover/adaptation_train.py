"""Bounded, reproducible engineering pilot; no checkpoint selection or tuning."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from .adaptation import (AdaptationPolicy, ObservationHistory, VARIANTS,
                         load_policy, make_env)
from .geometry import reactive_actions
from .policy import state_digest, update_policy
from .policy_train import terminal_mission_reward


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def line(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


def autonomous_mask(env):
    return torch.from_numpy(env.takeover_flags == 0)


class EpisodeMeter:
    def __init__(self):
        self.lag = {phase: 0.0 for phase in ('patrol', 'takeover', 'handback')}
        self.seconds = dict(self.lag)
        self.operator_seconds = 0.0
        self.onsets = 0
        self.phase = 'patrol'

    def before(self, env):
        self.before_time = env.time
        self.before_lag = np.maximum(env.slot_targets[:, 0] - env.positions[:, 0], 0)
        self.pre_phase = env.phase
        if env.phase == 'takeover' and self.phase != 'takeover':
            self.onsets += 1
        self.phase = env.phase

    def after(self, env):
        dt = env.time - self.before_time
        now = np.maximum(env.slot_targets[:, 0] - env.positions[:, 0], 0)
        self.lag[self.pre_phase] += float((now + self.before_lag).mean() * 0.5 * dt)
        self.seconds[self.pre_phase] += dt
        if self.pre_phase == 'takeover':
            self.operator_seconds += dt

    def result(self, env):
        metrics = env.metrics()
        return {**metrics, 'scenario_success': bool(env.check_completed and env.recovery_success and not env.collision),
            'failure_inclusive_recovery_cost': (float(env.recovery_time)
                if env.recovery_success and not env.collision else env.config.recovery_timeout),
            'collision_phase': env.collision_phase, 'collision_time': env.collision_time,
            'phase_min_distance': env.phase_min_distance, 'phase_lag_integral_m_s': self.lag,
            'phase_observation_seconds': self.seconds, 'operator_control_seconds': self.operator_seconds,
            'takeover_count': self.onsets, 'integrated_patrol_lag': sum(self.lag.values())}


def training_case(index, spec):
    cases = len(spec['geometries']) * len(spec['training_profiles'])
    geometry = spec['geometries'][index % len(spec['geometries'])]
    profile = spec['training_profiles'][(index // len(spec['geometries'])) % len(spec['training_profiles'])]
    return spec['training_scenario_seed_start'] + index // cases, geometry, profile


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Pilot wall-clock cap reached; partial data retained')


def train(spec, variant, seed, output, deadline):
    torch.manual_seed(seed)
    model = AdaptationPolicy(spec['frames'])
    initial = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'])
    action_rng = torch.Generator().manual_seed(seed)
    entropy_rng = torch.Generator().manual_seed(seed + 1000)
    shuffle_rng = np.random.default_rng(seed)
    history = ObservationHistory(variant, spec['frames'])
    index, steps, updates, physics = 0, 0, 0, 0

    def reset(index):
        scenario_seed, geometry, profile = training_case(index, spec)
        env = make_env(geometry, profile)
        obs, _ = env.reset(seed=scenario_seed)
        return env, history.reset(obs), EpisodeMeter(), (scenario_seed, geometry, profile)

    env, obs, meter, case = reset(index)
    started = time.monotonic()
    with gzip.open(output / 'training_transitions.jsonl.gz', 'wt', compresslevel=1) as transitions, \
            (output / 'training_episodes.jsonl').open('w') as episodes, \
            (output / 'training_updates.jsonl').open('w') as update_log:
        while steps < spec['training_steps']:
            rollout = {key: [] for key in ('observations', 'latents', 'log_probs',
                'masks', 'rewards', 'values', 'episode_ends')}
            for _ in range(min(spec['rollout_timesteps'], spec['training_steps'] - steps)):
                check_deadline(deadline)
                with torch.no_grad():
                    actions, latent, log_prob, mask, value = model.sample(
                        torch.from_numpy(obs), autonomous_mask(env), action_rng)
                meter.before(env)
                previous_physics = env.physics_steps
                raw, env_reward, terminated, truncated, info = env.step(actions.numpy())
                meter.after(env)
                done = terminated or truncated
                reward = env_reward + terminal_mission_reward(info, done)
                steps += 1
                physics += env.physics_steps - previous_physics
                line(transitions, {'step': steps, 'episode_index': index, 'case': case,
                    'observations': obs.tolist(), 'latent': latent.tolist(),
                    'log_prob': log_prob.tolist(), 'actions': actions.tolist(),
                    'autonomous_mask': mask.tolist(), 'value': float(value),
                    'reward': reward, 'environment_reward': env_reward,
                    'pre_phase': meter.pre_phase, 'time': env.time, 'phase': env.phase,
                    'terminated': terminated, 'truncated': truncated,
                    'requested_velocity': env.last_requested_velocity.tolist(),
                    'executed_velocity': env.last_executed_velocity.tolist()})
                for key, data in [('observations', obs.copy()), ('latents', latent.numpy()),
                    ('log_probs', log_prob.numpy()), ('masks', mask.numpy()), ('rewards', reward),
                    ('values', float(value)), ('episode_ends', done)]:
                    rollout[key].append(data)
                obs = history.push(raw)
                if done:
                    line(episodes, {'episode_index': index, 'case': case, **meter.result(env)})
                    episodes.flush()
                    index += 1
                    if steps < spec['training_steps']:
                        env, obs, meter, case = reset(index)
            with torch.no_grad():
                rollout['bootstrap'] = 0.0 if rollout['episode_ends'][-1] else float(model.value(torch.from_numpy(obs)))
            report = update_policy(model, optimizer, rollout, spec, shuffle_rng, entropy_rng)
            updates += 1
            progress = {'steps': steps, 'physics_steps': physics, 'updates': updates,
                'completed_episodes': index, 'wall_seconds': time.monotonic() - started}
            line(update_log, {**progress, **report})
            update_log.flush()
            transitions.flush()
            write(output / 'progress.json', progress)
            print(json.dumps({'variant': variant, **progress}), flush=True)
    write(output / 'training_last_episode.json', {'incomplete': not (env.terminated or env.truncated),
        'episode_index': index, 'case': case, **meter.result(env)})
    torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'frames': spec['frames'], 'variant': variant, 'seed': seed, 'specification': spec,
        'steps': steps, 'initial_sha256': initial, 'final_sha256': state_digest(model),
        'action_rng_state': action_rng.get_state(), 'entropy_rng_state': entropy_rng.get_state(),
        'shuffle_rng_state': shuffle_rng.bit_generator.state}, output / 'model.pt')
    restored, _ = load_policy(output / 'model.pt')
    if initial == state_digest(model) or state_digest(restored) != state_digest(model):
        raise RuntimeError('No learning or checkpoint reload mismatch')
    write(output / 'training_summary.json', {**progress, 'initial_sha256': initial,
        'final_sha256': state_digest(model), 'reload_identical': True})
    return restored


def evaluation_case(env, variant, model, frames, stream, identity):
    raw, _ = env.reset(seed=identity['scenario_seed'])
    history = ObservationHistory(variant if model is not None else 'flag', frames)
    obs = history.reset(raw)
    meter = EpisodeMeter()
    while not (env.terminated or env.truncated):
        if model is None:
            actions = (np.zeros((4, 3), dtype=np.float32) if variant == 'nominal'
                       else reactive_actions(raw, residual_velocity=env.config.residual_velocity))
            actions[env.takeover_flags.astype(bool)] = 0
        else:
            with torch.no_grad():
                actions = model.deterministic(torch.from_numpy(obs), autonomous_mask(env)).numpy()
        meter.before(env)
        raw, _, _, _, _ = env.step(actions)
        meter.after(env)
        obs = history.push(raw)
        line(stream, {**identity, 'time': env.time, 'pre_phase': meter.pre_phase,
            'phase': env.phase, 'positions': env.positions.tolist(), 'velocities': env.velocities.tolist(),
            'actions': actions.tolist(), 'requested_velocity': env.last_requested_velocity.tolist(),
            'executed_velocity': env.last_executed_velocity.tolist(),
            'override_mask': env.last_override_mask.tolist(), 'collision': env.collision})
    return {**identity, **meter.result(env)}


def summarize(cases):
    rows = []
    for policy, geometry, profile in sorted(set((c['policy'], c['geometry'], c['profile']) for c in cases)):
        selected = [c for c in cases if (c['policy'], c['geometry'], c['profile']) == (policy, geometry, profile)]
        mean = lambda key: float(np.mean([c[key] for c in selected]))
        rows.append({'policy': policy, 'geometry': geometry, 'profile': profile, 'n': len(selected),
            'success_rate': mean('scenario_success'), 'collision_rate': mean('collision'),
            'inspection_rate': mean('check_completed'), 'handback_observed_rate': mean('handover_observed'),
            'recovery_success_rate': mean('recovery_success'),
            'failure_inclusive_recovery_cost': mean('failure_inclusive_recovery_cost'),
            'takeover_collision_rate': float(np.mean([c['collision_phase'] == 'takeover' for c in selected])),
            'handback_collision_rate': float(np.mean([c['collision_phase'] == 'handback' for c in selected]))})
    return rows


def evaluate(spec, variant, model, output, deadline, include_rules):
    cases = []
    policies = [(variant, model)] + ([('nominal', None), ('reactive', None)] if include_rules else [])
    with gzip.open(output / 'evaluation_trajectories.jsonl.gz', 'wt', compresslevel=1) as stream:
        for policy, current in policies:
            for seed in spec['evaluation_scenario_seeds']:
                for geometry in spec['geometries']:
                    for profile in spec['evaluation_profiles']:
                        check_deadline(deadline)
                        identity = {'policy': policy, 'scenario_seed': seed, 'geometry': geometry,
                                    'profile': profile, 'split': 'pilot_evaluation'}
                        cases.append(evaluation_case(make_env(geometry, profile), policy, current,
                                                     spec['frames'], stream, identity))
                        write(output / 'evaluation_cases.json', cases)
    write(output / 'evaluation_summary.json', summarize(cases))


def run(args):
    spec = json.loads(Path(args.config).read_text())
    if args.smoke:
        spec.update(training_steps=256, rollout_timesteps=128, epochs=1,
                    evaluation_scenario_seeds=[59999], evaluation_profiles=['steady'], max_wall_seconds=180)
    if args.seed < 0 or spec['training_steps'] <= 0:
        raise ValueError('Invalid budget or seed')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    git = lambda *cmd: subprocess.check_output(['git', '-C', str(root), *cmd], text=True).strip()
    metadata = {'status': 'running', 'variant': args.variant, 'training_seed': args.seed,
        'source_commit': git('rev-parse', 'HEAD'), 'source_status': git('status', '--porcelain'),
        'engineering_smoke': args.smoke, 'torch': torch.__version__, 'numpy': np.__version__,
        'device': 'cpu', 'limits': ['program-simulated operator; no human evidence',
        'single-seed small-budget pilot; not convergence or publication evidence'],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py')}}
    write(output / 'metadata.json', metadata)
    write(output / 'effective_config.json', spec)
    write(output / 'environment_example.json', asdict(make_env('crossing', 'steady').config))
    try:
        if metadata['source_status'] and not args.smoke:
            raise RuntimeError('Commit source before a non-smoke pilot')
        model = train(spec, args.variant, args.seed, output, started + spec['max_wall_seconds'])
        evaluate(spec, args.variant, model, output, started + spec['max_wall_seconds'], args.include_rules)
        metadata['status'] = 'complete'
    except Exception:
        metadata.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic() - started
        write(output / 'metadata.json', metadata)
    write(output / 'artifact_hashes.json', {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/takeover_adaptation_pilot.json')
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--include-rules', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
