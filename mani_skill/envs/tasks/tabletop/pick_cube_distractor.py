import math

import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeDistractor-v1", max_episode_steps=50)
class PickCubeDistractorEnv(PickCubeEnv):
    """PickCube with visually distinct, task-irrelevant cubes.

    The red cube inherited from :class:`PickCubeEnv` remains the only target.
    Distractors are physical actors, but they are intentionally excluded from
    observations, success conditions, and rewards other than through RGB pixels.
    """

    DISTRACTOR_COLORS = (
        [0, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
    )
    DISTRACTOR_CLEARANCE = 0.005
    MAX_PLACEMENT_TRIALS = 1000

    def __init__(self, *args, num_distractors: int = 3, **kwargs):
        if (
            isinstance(num_distractors, bool)
            or not isinstance(num_distractors, int)
            or num_distractors not in range(4)
        ):
            raise ValueError("num_distractors must be one of 0, 1, 2, or 3")
        self.num_distractors = num_distractors
        self.distractors = []
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.distractors = []
        for i in range(self.num_distractors):
            distractor = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.DISTRACTOR_COLORS[i],
                name=f"distractor_cube_{i}",
                initial_pose=sapien.Pose(
                    p=[1 + i * self.cube_half_size * 3, 0, self.cube_half_size]
                ),
            )
            self.distractors.append(distractor)

    def _sample_distractor_xy(self, env_idx: torch.Tensor) -> list[torch.Tensor]:
        """Sample collision-free distractor centers for the resetting sub-scenes."""
        batch_size = len(env_idx)
        spawn_half_size = self.cube_spawn_half_size * 1.5
        spawn_center = torch.tensor(
            self.cube_spawn_center, dtype=torch.float32, device=self.device
        )
        bounds_low = spawn_center - spawn_half_size
        bounds_range = torch.full(
            (2,), spawn_half_size * 2, dtype=torch.float32, device=self.device
        )

        fixture_positions = [
            self.cube.pose.p[env_idx, :2],
            self.goal_site.pose.p[env_idx, :2],
        ]
        cube_xy_radius = math.sqrt(2) * self.cube_half_size
        fixture_radii = [cube_xy_radius, self.goal_thresh]
        sampled_positions = []

        for distractor_index in range(self.num_distractors):
            sampled_xy = torch.zeros(
                (batch_size, 2), dtype=torch.float32, device=self.device
            )
            remaining = torch.ones(
                batch_size, dtype=torch.bool, device=self.device
            )
            required_distances = torch.tensor(
                [
                    cube_xy_radius + fixture_radius + self.DISTRACTOR_CLEARANCE
                    for fixture_radius in fixture_radii
                ],
                dtype=torch.float32,
                device=self.device,
            )

            for _ in range(self.MAX_PLACEMENT_TRIALS):
                candidates = (
                    torch.rand((batch_size, 2), device=self.device) * bounds_range
                    + bounds_low
                )
                distances = torch.stack(
                    [
                        torch.linalg.norm(candidates - fixture_position, dim=1)
                        for fixture_position in fixture_positions
                    ]
                )
                valid = torch.all(
                    distances > required_distances[:, None], dim=0
                ) & remaining
                sampled_xy[valid] = candidates[valid]
                remaining[valid] = False
                if not torch.any(remaining):
                    break

            if torch.any(remaining):
                failed_count = int(remaining.sum().item())
                raise RuntimeError(
                    "Failed to place distractor "
                    f"{distractor_index} in {failed_count} parallel environment(s) "
                    f"after {self.MAX_PLACEMENT_TRIALS} trials"
                )

            sampled_positions.append(sampled_xy)
            fixture_positions.append(sampled_xy)
            fixture_radii.append(cube_xy_radius)

        return sampled_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Calling the parent first preserves PickCube's target/goal RNG sequence,
        # especially when num_distractors is zero.
        super()._initialize_episode(env_idx, options)
        if self.num_distractors == 0:
            return

        batch_size = len(env_idx)
        distractor_positions = self._sample_distractor_xy(env_idx)
        for distractor, xy in zip(self.distractors, distractor_positions):
            xyz = torch.zeros(
                (batch_size, 3), dtype=torch.float32, device=self.device
            )
            xyz[:, :2] = xy
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(
                batch_size, lock_x=True, lock_y=True
            )
            distractor.set_pose(Pose.create_from_pq(xyz, qs))
