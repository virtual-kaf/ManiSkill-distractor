import torch

from examples.baselines.ppo.vcd import (
    apply_plausibility_constraint,
    contrastive_action_mean,
    distort_rgb_observation,
    make_generator,
)


def test_contrastive_degeneracy_cases():
    clean = torch.tensor([[0.1, -0.2]])
    distorted = torch.tensor([[0.8, 0.4]])
    torch.testing.assert_close(
        contrastive_action_mean(clean, distorted, alpha=0), clean
    )
    torch.testing.assert_close(
        contrastive_action_mean(clean, clean, alpha=3), clean
    )


def test_beta_one_collapses_to_clean_mean():
    clean = torch.tensor([[0.1, -0.2]])
    contrastive = torch.tensor([[2.0, -3.0]])
    std = torch.tensor([[0.5, 0.5]])
    constrained = apply_plausibility_constraint(
        clean, contrastive, std, beta=1
    )
    torch.testing.assert_close(constrained, clean)


def test_plausibility_projection_respects_mahalanobis_radius():
    clean = torch.zeros((2, 3))
    contrastive = torch.tensor([[10.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    std = torch.tensor([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])
    beta = 0.1
    constrained = apply_plausibility_constraint(
        clean, contrastive, std, beta=beta
    )
    norms = torch.linalg.vector_norm((constrained - clean) / std, dim=-1)
    assert torch.all(norms <= (-2 * torch.log(torch.tensor(beta))).sqrt() + 1e-6)


def test_beta_zero_does_not_clip_action():
    clean = torch.zeros((1, 2))
    contrastive = torch.tensor([[4.0, -5.0]])
    constrained = apply_plausibility_constraint(
        clean, contrastive, torch.ones_like(clean), beta=0
    )
    torch.testing.assert_close(constrained, contrastive)
    assert torch.any(torch.abs(constrained) > 1)


def test_rgb_distortion_is_reproducible_and_does_not_mutate_observation():
    observation = {
        "rgb": torch.full((2, 4, 4, 3), 128, dtype=torch.uint8),
        "state": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    original_rgb = observation["rgb"].clone()
    first = distort_rgb_observation(
        observation,
        gamma=0.1,
        noise_steps=500,
        generator=make_generator(torch.device("cpu"), 123),
    )
    second = distort_rgb_observation(
        observation,
        gamma=0.1,
        noise_steps=500,
        generator=make_generator(torch.device("cpu"), 123),
    )

    torch.testing.assert_close(first["rgb"], second["rgb"])
    torch.testing.assert_close(observation["rgb"], original_rgb)
    assert first["state"] is observation["state"]
    assert first["rgb"] is not observation["rgb"]
