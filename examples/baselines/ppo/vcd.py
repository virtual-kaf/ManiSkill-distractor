"""Continuous-action Visual Contrastive Decoding for the RGB PPO baseline.

The original VCD method contrasts categorical output logits. The ManiSkill PPO
policy instead emits a Normal distribution with a learned, observation-independent
standard deviation. Contrasting the two Normal log densities therefore produces
another Normal distribution with the same standard deviation and mean

    (1 + alpha) * clean_mean - alpha * distorted_mean.

No function in this module clips actions. Vanilla and VCD actions must both be
passed directly to ``env.step`` so ManiSkill applies its existing controller-level
action clipping uniformly.
"""

import math
from typing import Protocol

import torch


class GaussianPolicy(Protocol):
    actor_logstd: torch.Tensor

    def get_action(self, observation: dict, deterministic: bool = False): ...


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a device-local generator without changing global RNG state."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def distort_rgb_observation(
    observation: dict[str, torch.Tensor],
    *,
    gamma: float,
    noise_steps: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Return a shallow observation copy with diffusion-noised RGB pixels.

    The PPO encoder divides its RGB input by 255. We apply the diffusion process
    in that normalized space and multiply by 255 before returning, allowing the
    unchanged encoder to consume the distorted floating-point tensor.
    """
    if "rgb" not in observation:
        raise KeyError("VCD requires an observation with an 'rgb' tensor")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if (
        isinstance(noise_steps, bool)
        or not isinstance(noise_steps, int)
        or noise_steps < 0
    ):
        raise ValueError("noise_steps must be a non-negative integer")

    rgb = observation["rgb"].float() / 255.0
    retained_variance = (1.0 - gamma) ** noise_steps
    noise = torch.randn(
        rgb.shape,
        dtype=rgb.dtype,
        device=rgb.device,
        generator=generator,
    )
    distorted_rgb = (
        math.sqrt(retained_variance) * rgb
        + math.sqrt(1.0 - retained_variance) * noise
    )

    distorted_observation = dict(observation)
    distorted_observation["rgb"] = distorted_rgb * 255.0
    return distorted_observation


def contrastive_action_mean(
    clean_mean: torch.Tensor,
    distorted_mean: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return (1.0 + alpha) * clean_mean - alpha * distorted_mean


def apply_plausibility_constraint(
    clean_mean: torch.Tensor,
    contrastive_mean: torch.Tensor,
    action_std: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Project a contrastive mean into the clean policy's density level set.

    For beta in (0, 1], the allowed Mahalanobis radius is
    ``sqrt(-2 * log(beta))``. Beta zero disables the constraint. In particular,
    beta one has a zero-radius region and returns the clean mean.
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must satisfy 0 <= beta <= 1")
    if beta == 0:
        return contrastive_mean
    if torch.any(action_std <= 0):
        raise ValueError("action_std must be strictly positive")

    delta = contrastive_mean - clean_mean
    mahalanobis_norm = torch.linalg.vector_norm(delta / action_std, dim=-1)
    radius = math.sqrt(-2.0 * math.log(beta))
    scale = torch.clamp(
        radius / torch.clamp(mahalanobis_norm, min=torch.finfo(delta.dtype).eps),
        max=1.0,
    )
    return clean_mean + delta * scale.unsqueeze(-1)


def get_vcd_action(
    policy: GaussianPolicy,
    observation: dict[str, torch.Tensor],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    noise_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Compute an unclipped deterministic continuous-action VCD action."""
    clean_mean = policy.get_action(observation, deterministic=True)
    distorted_observation = distort_rgb_observation(
        observation,
        gamma=gamma,
        noise_steps=noise_steps,
        generator=generator,
    )
    distorted_mean = policy.get_action(distorted_observation, deterministic=True)
    contrastive_mean = contrastive_action_mean(
        clean_mean, distorted_mean, alpha=alpha
    )
    action_std = policy.actor_logstd.exp().expand_as(clean_mean)
    return apply_plausibility_constraint(
        clean_mean,
        contrastive_mean,
        action_std,
        beta=beta,
    )
