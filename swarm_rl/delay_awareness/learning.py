"""Current-observation and finite-window GRU PPO baselines on delayed messages.

The GRU re-encodes eight causal frames (0.7 seconds); no hidden state persists
beyond that window. This is a bounded baseline, not a new awareness method.
"""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from swarm_rl.takeover.policy import state_digest, squashed_log_prob, update_policy
from .core import DelayedChannel, RouteEnv
from .diagnose import aggregate, fingerprint, line, run_case, source_record, write


class History:
    def __init__(self, frames=8):
        self.frames = frames
        self.buffer = deque(maxlen=frames)

    def push(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (4, 36):
            raise ValueError('Expected local message observations')
        self.buffer.append(obs.copy())
        return np.concatenate([np.zeros_like(obs)] * (self.frames - len(self.buffer)) + list(self.buffer), axis=-1)


class Policy(nn.Module):
    def __init__(self, variant, frames=8):
        super().__init__()
        if variant not in ('mlp', 'gru'):
            raise ValueError(variant)
        self.variant, self.frames = variant, frames
        scale = np.ones(36, dtype=np.float32)
        scale[:12] = [1.2]*3 + [6.0]*3 + [1.2]*6
        for start in (12, 20, 28):
            scale[start:start+6] = [6.0]*3 + [2.4]*3
        self.register_buffer('scales', torch.from_numpy(scale))
        if variant == 'gru':
            self.encoder = nn.GRU(36, 64, batch_first=True)
            self.actor = nn.Linear(64, 3)
        else:
            # Approximate actor capacity match: 19523 vs GRU 19779 parameters.
            self.actor = nn.Sequential(nn.Linear(36, 128), nn.Tanh(), nn.Linear(128, 112),
                                       nn.Tanh(), nn.Linear(112, 3))
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        self.critic = nn.Sequential(nn.Linear(144, 128), nn.Tanh(), nn.Linear(128, 128),
                                    nn.Tanh(), nn.Linear(128, 1))

    def sequence(self, obs):
        if obs.shape[-1] != 36 * self.frames:
            raise ValueError('Wrong history width')
        return (obs.reshape(*obs.shape[:-1], self.frames, 36) / self.scales).clamp(-10, 10)

    def distribution(self, obs):
        sequence = self.sequence(obs)
        if self.variant == 'gru':
            _, hidden = self.encoder(sequence.reshape(-1, self.frames, 36))
            mean = self.actor(hidden[-1]).reshape(*obs.shape[:-1], 3)
        else:
            mean = self.actor(sequence[..., -1, :])
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        if obs.shape[-2] != 4:
            raise ValueError('Central critic needs four observations')
        current = self.sequence(obs)[..., -1, :]
        return self.critic(current.flatten(-2)).squeeze(-1)

    def sample(self, obs, generator):
        distribution = self.distribution(obs)
        latent = distribution.mean + distribution.stddev * torch.randn(distribution.mean.shape, generator=generator)
        return latent.tanh(), latent, squashed_log_prob(distribution, latent), self.value(obs)


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = Policy(saved['variant'], saved['frames'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved


def train(spec, variant, output, deadline):
    seed = spec['training_seed']
    torch.manual_seed(seed)
    model = Policy(variant, spec['history_frames'])
    initial = state_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'])
    action_rng = torch.Generator().manual_seed(seed)
    entropy_rng = torch.Generator().manual_seed(seed + 1000)
    shuffle_rng = np.random.default_rng(seed)

    def reset(index):
        # Shared seeded case schedule; equal control-step budget, not equal episode count.
        geometry = spec['geometries'][index % len(spec['geometries'])]
        delay = spec['training_delay_steps'][(index // len(spec['geometries'])) % len(spec['training_delay_steps'])]
        scenario_seed = spec['training_scenario_seed_start'] + index // (len(spec['geometries']) * len(spec['training_delay_steps']))
        env = RouteEnv(geometry)
        env.reset(seed=scenario_seed)
        channel, history = DelayedChannel(delay), History(spec['history_frames'])
        obs = history.push(channel.observation(env)[0])
        return env, channel, history, obs, {'geometry': geometry, 'delay_steps': delay, 'scenario_seed': scenario_seed}

    index, steps, updates = 0, 0, 0
    env, channel, history, obs, case = reset(index)
    started = time.monotonic()
    with gzip.open(output / 'training_transitions.jsonl.gz', 'wt', compresslevel=1) as transitions, \
            (output / 'training_episodes.jsonl').open('w') as episodes, \
            (output / 'training_updates.jsonl').open('w') as update_log:
        while steps < spec['training_steps']:
            rollout = {k: [] for k in ('observations', 'latents', 'log_probs', 'masks',
                                     'rewards', 'values', 'episode_ends')}
            for _ in range(min(spec['rollout_timesteps'], spec['training_steps'] - steps)):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Learning baseline wall cap reached')
                with torch.no_grad():
                    action, latent, log_probs, value = model.sample(torch.from_numpy(obs), action_rng)
                _, reward, terminated, truncated, metrics = env.step(action.numpy())
                done = terminated or truncated
                mask = np.ones(4, dtype=bool)
                for key, data in [('observations', obs.copy()), ('latents', latent.numpy()),
                    ('log_probs', log_probs.numpy()), ('masks', mask), ('rewards', reward),
                    ('values', float(value)), ('episode_ends', done)]:
                    rollout[key].append(data)
                steps += 1
                line(transitions, {'step': steps, 'episode_index': index, **case,
                    'observations': obs.tolist(), 'actions': action.tolist(), 'latent': latent.tolist(),
                    'log_probs': log_probs.tolist(), 'value': float(value), 'reward': reward,
                    'terminated': terminated, 'truncated': truncated})
                obs = history.push(channel.observation(env)[0])
                if done:
                    line(episodes, {'episode_index': index, **case, **metrics})
                    episodes.flush()
                    index += 1
                    if steps < spec['training_steps']:
                        env, channel, history, obs, case = reset(index)
            with torch.no_grad():
                rollout['bootstrap'] = 0.0 if rollout['episode_ends'][-1] else float(model.value(torch.from_numpy(obs)))
            report = update_policy(model, optimizer, rollout, spec, shuffle_rng, entropy_rng)
            updates += 1
            progress = {'steps': steps, 'updates': updates, 'completed_episodes': index,
                        'wall_seconds': time.monotonic() - started}
            line(update_log, {**progress, **report})
            update_log.flush()
            write(output / 'progress.json', progress)
            print(json.dumps({'variant': variant, **progress}), flush=True)
    write(output / 'last_training_episode.json', {'incomplete': not (env.terminated or env.truncated),
        **case, **env.route_metrics()})
    torch.save({'state_dict': model.state_dict(), 'variant': variant, 'frames': spec['history_frames'],
        'specification': spec, 'steps': steps, 'optimizer_state_dict': optimizer.state_dict(),
        'action_rng_state': action_rng.get_state(), 'entropy_rng_state': entropy_rng.get_state(),
        'shuffle_rng_state': shuffle_rng.bit_generator.state}, output / 'model.pt')
    restored, _ = load(output / 'model.pt')
    if initial == state_digest(model) or state_digest(restored) != state_digest(model):
        raise RuntimeError('Unchanged model or failed checkpoint identity check')
    write(output / 'training_summary.json', {**progress, 'initial_sha256': initial,
        'final_sha256': state_digest(model), 'reload_identical': True,
        'actor_parameters': sum(p.numel() for name, p in model.named_parameters() if not name.startswith('critic')),
        'critic_parameters': sum(p.numel() for name, p in model.named_parameters() if name.startswith('critic'))})
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/delay_awareness_pilot.json'))
    parser.add_argument('--variant', choices=('mlp', 'gru'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if args.smoke:
        spec.update(training_steps=256, rollout_timesteps=128, epochs=1,
                    scenario_seeds=[71998], delay_steps=[0, 2])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    (output / 'trajectories').mkdir()
    torch.set_num_threads(1)
    metadata = {**source_record(output), 'status': 'running', 'variant': args.variant,
        'smoke': args.smoke, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
        'claim': 'single-seed bounded learning baseline; no convergence or novelty claim'}
    write(output / 'metadata.json', metadata)
    write(output / 'effective_config.json', spec)
    started = time.monotonic()
    deadline = started + spec['max_wall_seconds']
    try:
        if metadata['source_status'] and not args.smoke:
            raise RuntimeError('Commit source before learning baseline')
        model = train(spec, args.variant, output, deadline)
        cases = []
        for geometry in spec['geometries']:
            for delay in spec['delay_steps']:
                for seed in spec['scenario_seeds']:
                    case = {'policy': args.variant, 'geometry': geometry, 'delay_steps': delay, 'scenario_seed': seed}
                    cases.append(run_case(case, output, deadline, learned=model))
                    write(output / 'all_cases.json', cases)
        write(output / 'summary.json', aggregate(cases))
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
