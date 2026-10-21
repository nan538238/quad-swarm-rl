"""CP-C2-01 shared row-local actor and separate centralized team critic.

Rollout storage retains the unsquashed Gaussian sample: saturated float32 tanh
actions cannot be inverted reliably. Each active teammate has its own PPO ratio;
the common team advantage is broadcast, never a product of four action ratios.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def observation_scale():
    scale = np.ones(41, dtype=np.float32)
    for start in (0, 9, 15, 22, 29, 38):
        scale[start:start + 3] = 1.2
    for start in (3, 6, 12, 19, 26):
        scale[start:start + 3] = 3.0
    scale[34] = 5.0
    return scale


def actor_mask(observations):
    """Mask uses only pre-step current flags and phase from the actor input."""
    return (observations[..., 36] > 0.5) & (observations[..., 33] < 0.5)


def squashed_log_prob(distribution, latent):
    # log(1-tanh(z)^2), stable even when tanh(z) rounds to +/-1.
    correction = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
    return (distribution.log_prob(latent) - correction).sum(-1)


class TeamPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('observation_scale', torch.from_numpy(observation_scale()))
        self.actor = nn.Sequential(nn.Linear(41, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        self.critic = nn.Sequential(nn.Linear(164, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def normalize(self, observations):
        if observations.shape[-1] != 41:
            raise ValueError('Actor input must have 41 current-state values')
        return (observations / self.observation_scale).clamp(-10.0, 10.0)

    def distribution(self, observations):
        means = self.actor(self.normalize(observations))
        return torch.distributions.Normal(means, self.log_std.exp().expand_as(means))

    def value(self, observations):
        if observations.shape[-2:] != (4, 41):
            raise ValueError('Team critic requires four 41-value current observations')
        return self.critic(self.normalize(observations).flatten(-2)).squeeze(-1)

    def deterministic(self, observations):
        actions = self.distribution(observations).mean.tanh()
        return torch.where(actor_mask(observations).unsqueeze(-1), actions,
                           torch.zeros_like(actions))

    def sample(self, observations, generator):
        distribution = self.distribution(observations)
        noise = torch.randn(distribution.mean.shape, generator=generator,
                            dtype=observations.dtype, device=observations.device)
        latent = distribution.mean + distribution.stddev * noise
        mask = actor_mask(observations)
        actions = torch.where(mask.unsqueeze(-1), latent.tanh(), torch.zeros_like(latent))
        return actions, latent, squashed_log_prob(distribution, latent), mask, self.value(observations)


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generalized_advantages(rewards, values, episode_ends, bootstrap,
                           gamma=0.995, gae_lambda=0.95):
    """Actual terminal *or 24s truncation* stops GAE; rollout cuts bootstrap."""
    rewards, values = np.asarray(rewards), np.asarray(values)
    ends = np.asarray(episode_ends, dtype=bool)
    if rewards.ndim != 1 or rewards.shape != values.shape or ends.shape != rewards.shape:
        raise ValueError('GAE expects equal one-dimensional arrays')
    advantage = np.zeros_like(rewards, dtype=np.float32)
    next_value, carry = float(bootstrap), 0.0
    for index in reversed(range(len(rewards))):
        alive = 1.0 - float(ends[index])
        delta = rewards[index] + gamma * next_value * alive - values[index]
        carry = delta + gamma * gae_lambda * alive * carry
        advantage[index] = carry
        next_value = values[index]
    return advantage, advantage + values


def ppo_actor_loss(log_probs, old_log_probs, advantages, mask, clip=0.2):
    if not mask.any():
        return log_probs.sum() * 0.0, {'approx_kl': 0.0, 'clip_fraction': 0.0}
    difference = log_probs[mask] - old_log_probs[mask]
    ratio = difference.exp()
    active_advantages = advantages[:, None].expand_as(mask)[mask]
    loss = -torch.minimum(ratio * active_advantages,
                          ratio.clamp(1.0 - clip, 1.0 + clip) * active_advantages).mean()
    return loss, {'approx_kl': float(((ratio - 1.0) - difference).mean().detach()),
                  'clip_fraction': float(((ratio - 1.0).abs() > clip).float().mean().detach())}


def update_policy(model, optimizer, rollout, specification, shuffle_rng, entropy_rng):
    obs = torch.as_tensor(np.asarray(rollout['observations']), dtype=torch.float32)
    latent = torch.as_tensor(np.asarray(rollout['latents']), dtype=torch.float32)
    old_log = torch.as_tensor(np.asarray(rollout['log_probs']), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(rollout['masks']), dtype=torch.bool)
    advantages, returns = generalized_advantages(
        rollout['rewards'], rollout['values'], rollout['episode_ends'],
        rollout['bootstrap'], specification['gamma'], specification['gae_lambda'])
    adv = torch.as_tensor(advantages, dtype=torch.float32)
    targets = torch.as_tensor(returns, dtype=torch.float32)
    active = adv[:, None].expand_as(mask)[mask]
    if active.numel():
        adv = (adv - active.mean()) / (active.std(unbiased=False) + 1e-8)
    before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    rows = []
    for epoch in range(specification['epochs']):
        for indices in np.array_split(shuffle_rng.permutation(len(obs)),
                                      math.ceil(len(obs) / specification['minibatch_timesteps'])):
            indices = torch.as_tensor(indices)
            distribution = model.distribution(obs[indices])
            log_probs = squashed_log_prob(distribution, latent[indices])
            actor_loss, stats = ppo_actor_loss(log_probs, old_log[indices], adv[indices],
                                              mask[indices], specification['clip'])
            active_mask = mask[indices]
            # MC differential entropy of the squashed distribution, with pathwise
            # gradients. Its independent generator cannot perturb collection noise.
            entropy_sample = distribution.mean + distribution.stddev * torch.randn(
                distribution.mean.shape, generator=entropy_rng)
            entropy_rows = -squashed_log_prob(distribution, entropy_sample)
            entropy = (entropy_rows[active_mask].mean() if active_mask.any()
                       else entropy_rows.sum() * 0.0)
            value_loss = (model.value(obs[indices]) - targets[indices]).square().mean()
            loss = (actor_loss + specification['value_coefficient'] * value_loss
                    - specification['entropy_coefficient'] * entropy)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           specification['max_grad_norm'])
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError('Nonfinite PPO gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite PPO parameters')
            rows.append({'epoch': epoch, 'timesteps': len(indices),
                         'active_actor_rows': int(active_mask.sum()),
                         'actor_loss': float(actor_loss.detach()),
                         'value_loss': float(value_loss.detach()), 'entropy': float(entropy.detach()),
                         'gradient_norm_before_clipping': float(gradient_norm), **stats})
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    return {'minibatches': rows, 'parameter_change_l2': float((after - before).norm()),
            'active_actor_rows': int(mask.sum()), 'rollout_timesteps': len(obs),
            'bootstrap_value': float(rollout['bootstrap']),
            'last_transition_actual_episode_end': bool(rollout['episode_ends'][-1]),
            'return_mean': float(targets.mean()), 'return_min': float(targets.min()),
            'return_max': float(targets.max())}


def load_policy(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = TeamPolicy()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
