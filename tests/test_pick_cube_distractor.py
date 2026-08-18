import math

import gymnasium as gym
import numpy as np
import pytest
import torch

import mani_skill.envs
from mani_skill.utils.structs.pose import Pose


@pytest.mark.parametrize("num_distractors", [0, 1, 2, 3])
def test_pick_cube_distractor_count_and_placement(num_distractors):
    env = gym.make(
        "PickCubeDistractor-v1",
        num_distractors=num_distractors,
        obs_mode="state_dict",
        sim_backend="physx_cpu",
    )
    base_env = env.unwrapped
    env.reset(seed=123)

    assert len(base_env.distractors) == num_distractors
    assert [actor.name for actor in base_env.distractors] == [
        f"distractor_cube_{i}" for i in range(num_distractors)
    ]
    assert base_env.DISTRACTOR_COLORS[:num_distractors] == (
        [0, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
    )[:num_distractors]

    cube_radius = math.sqrt(2) * base_env.cube_half_size
    fixture_positions = [base_env.cube.pose.p[0, :2], base_env.goal_site.pose.p[0, :2]]
    fixture_radii = [cube_radius, base_env.goal_thresh]
    for distractor in base_env.distractors:
        distractor_xy = distractor.pose.p[0, :2]
        for fixture_xy, fixture_radius in zip(fixture_positions, fixture_radii):
            distance = torch.linalg.norm(distractor_xy - fixture_xy).item()
            assert distance > (
                cube_radius + fixture_radius + base_env.DISTRACTOR_CLEARANCE
            )
        fixture_positions.append(distractor_xy)
        fixture_radii.append(cube_radius)

    env.close()


def test_distractor_reset_randomization_is_seeded():
    env = gym.make(
        "PickCubeDistractor-v1",
        num_distractors=3,
        obs_mode="state_dict",
        sim_backend="physx_cpu",
    )
    base_env = env.unwrapped
    env.reset(seed=321)
    first = torch.stack(
        [distractor.pose.raw_pose.clone() for distractor in base_env.distractors]
    )
    env.reset(seed=321)
    repeated = torch.stack(
        [distractor.pose.raw_pose.clone() for distractor in base_env.distractors]
    )
    env.reset(seed=322)
    changed = torch.stack(
        [distractor.pose.raw_pose.clone() for distractor in base_env.distractors]
    )

    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, changed)
    env.close()


@pytest.mark.parametrize("num_distractors", [-1, 4, True, 1.0])
def test_pick_cube_distractor_rejects_invalid_count(num_distractors):
    with pytest.raises(ValueError, match="num_distractors"):
        gym.make("PickCubeDistractor-v1", num_distractors=num_distractors)


def test_zero_distractors_matches_pick_cube_initialization():
    original_env = gym.make("PickCube-v1", obs_mode="state_dict", sim_backend="physx_cpu")
    distractor_env = gym.make(
        "PickCubeDistractor-v1",
        num_distractors=0,
        obs_mode="state_dict",
        sim_backend="physx_cpu",
    )
    original_env.reset(seed=456)
    distractor_env.reset(seed=456)

    np.testing.assert_allclose(
        original_env.unwrapped.cube.pose.raw_pose.cpu().numpy(),
        distractor_env.unwrapped.cube.pose.raw_pose.cpu().numpy(),
    )
    np.testing.assert_allclose(
        original_env.unwrapped.goal_site.pose.raw_pose.cpu().numpy(),
        distractor_env.unwrapped.goal_site.pose.raw_pose.cpu().numpy(),
    )

    original_env.close()
    distractor_env.close()


def test_distractors_do_not_change_task_semantics():
    env = gym.make(
        "PickCubeDistractor-v1",
        num_distractors=1,
        obs_mode="state_dict",
        sim_backend="physx_cpu",
    )
    base_env = env.unwrapped
    env.reset(seed=789)

    target_position = torch.tensor(
        [[-0.08, -0.08, base_env.cube_half_size]], device=base_env.device
    )
    goal_position = torch.tensor(
        [[0.08, 0.08, base_env.cube_half_size + 0.1]], device=base_env.device
    )
    base_env.cube.set_pose(Pose.create_from_pq(target_position))
    base_env.goal_site.set_pose(Pose.create_from_pq(goal_position))
    target_pose = base_env.cube.pose.raw_pose.clone()
    goal_pose = base_env.goal_site.pose.raw_pose.clone()
    base_env.distractors[0].set_pose(Pose.create_from_pq(goal_pose[:, :3]))
    info_at_goal = base_env.evaluate()
    assert not bool(info_at_goal["is_obj_placed"][0])
    assert not bool(info_at_goal["success"][0])

    action = torch.zeros(
        (1,) + base_env.single_action_space.shape, device=base_env.device
    )
    reward_at_goal = base_env.compute_dense_reward({}, action, info_at_goal).clone()
    far_pose = goal_pose.clone()
    far_pose[:, 0] += 0.2
    base_env.distractors[0].set_pose(Pose.create_from_pq(far_pose[:, :3]))
    base_env.cube.set_pose(Pose.create_from_pq(target_pose[:, :3], target_pose[:, 3:]))
    info_far = base_env.evaluate()
    reward_far = base_env.compute_dense_reward({}, action, info_far)

    torch.testing.assert_close(reward_at_goal, reward_far)
    assert info_at_goal.keys() == info_far.keys()
    assert all("distractor" not in key for key in base_env._get_obs_extra(info_far))

    env.close()
