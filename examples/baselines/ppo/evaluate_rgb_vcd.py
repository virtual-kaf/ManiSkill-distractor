"""Paired evaluation of vanilla RGB PPO and continuous-action VCD."""

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import tyro

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

try:
    from .ppo_rgb import Agent
    from .vcd import get_vcd_action, make_generator
except ImportError:
    from ppo_rgb import Agent
    from vcd import get_vcd_action, make_generator


@dataclass
class Args:
    checkpoint: str
    """path to a ppo_rgb.py state-dict checkpoint"""
    env_id: str = "PickCubeDistractor-v1"
    """environment ID; the default supports num_distractors"""
    distractor_counts: tuple[int, ...] = (0, 1, 2, 3)
    """distractor counts evaluated with the same checkpoint"""
    num_eval_envs: int = 8
    """number of parallel evaluation environments"""
    num_eval_episodes: int = 100
    """number of explicitly seeded episodes per mode and distractor count"""
    seed: int = 1
    """first environment seed; episode i uses seed + i"""
    vcd_seed: int = 10001
    """seed for the VCD diffusion-noise generator"""
    torch_deterministic: bool = True
    include_state: bool = True
    """match the ppo_rgb.py checkpoint's observation configuration"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    render_mode: str = "all"
    sim_backend: str = "physx_cuda"
    capture_video: bool = False
    output_dir: Optional[str] = None
    """defaults to CHECKPOINT_DIR/vcd_eval_TIMESTAMP"""
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.1
    noise_steps: int = 500


def validate_args(args: Args):
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.num_eval_envs <= 0:
        raise ValueError("num_eval_envs must be positive")
    if args.num_eval_episodes <= 0:
        raise ValueError("num_eval_episodes must be positive")
    if not args.distractor_counts:
        raise ValueError("distractor_counts cannot be empty")
    invalid_counts = [count for count in args.distractor_counts if count not in range(4)]
    if invalid_counts:
        raise ValueError(
            f"distractor_counts must contain only 0, 1, 2, or 3; got {invalid_counts}"
        )
    if args.sim_backend in {"physx_cpu", "sapien_cpu"} and args.num_eval_envs != 1:
        raise ValueError("CPU simulation requires num_eval_envs=1")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0 <= args.beta <= 1:
        raise ValueError("beta must satisfy 0 <= beta <= 1")
    if not 0 <= args.gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    if args.noise_steps < 0:
        raise ValueError("noise_steps must be non-negative")


def make_eval_env(args: Args, num_distractors: int, video_dir: Optional[Path]):
    env_kwargs = dict(
        obs_mode="rgb",
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        num_distractors=num_distractors,
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=0,
        **env_kwargs,
    )
    env = FlattenRGBDObservationWrapper(
        env, rgb=True, depth=False, state=args.include_state
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    max_episode_steps = gym_utils.find_max_episode_steps_value(env)
    if max_episode_steps is None:
        raise RuntimeError(f"{args.env_id} has no registered episode horizon")
    if video_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            save_trajectory=False,
            max_steps_per_video=max_episode_steps,
            video_fps=30,
        )
    env = ManiSkillVectorEnv(
        env,
        args.num_eval_envs,
        auto_reset=False,
        ignore_terminations=True,
        record_metrics=True,
    )
    assert isinstance(env.single_action_space, gym.spaces.Box)
    return env, max_episode_steps


def tensor_item(value: torch.Tensor, index: int):
    return value[index].detach().cpu().item()


def evaluate_setting(
    args: Args,
    *,
    mode: str,
    num_distractors: int,
    checkpoint_state: dict,
    output_dir: Path,
) -> list[dict]:
    video_dir = None
    if args.capture_video:
        video_dir = output_dir / "videos" / f"distractors_{num_distractors}" / mode
    env, horizon = make_eval_env(args, num_distractors, video_dir)
    rows = []
    agent = None
    noise_generator = None

    try:
        for batch_start in range(0, args.num_eval_episodes, args.num_eval_envs):
            episode_seeds = [
                args.seed + batch_start + env_index
                for env_index in range(args.num_eval_envs)
            ]
            observation, _ = env.reset(seed=episode_seeds)
            if agent is None:
                device = observation["rgb"].device
                agent = Agent(env, sample_obs=observation).to(device)
                agent.load_state_dict(checkpoint_state, strict=True)
                agent.eval()
                noise_generator = make_generator(device, args.vcd_seed)

            episode_info = None
            truncations = None
            for _ in range(horizon):
                with torch.no_grad():
                    if mode == "vanilla":
                        action = agent.get_action(observation, deterministic=True)
                    elif mode == "vcd":
                        action = get_vcd_action(
                            agent,
                            observation,
                            alpha=args.alpha,
                            beta=args.beta,
                            gamma=args.gamma,
                            noise_steps=args.noise_steps,
                            generator=noise_generator,
                        )
                    else:
                        raise ValueError(f"Unknown evaluation mode: {mode}")

                # Both modes deliberately use this exact call. ManiSkill performs
                # action-space clipping later in the shared controller path.
                observation, _, _, truncations, info = env.step(action)
                episode_info = info["episode"]

            if not bool(torch.all(truncations)):
                raise RuntimeError(
                    f"Expected all evaluation environments to truncate at {horizon} steps"
                )
            keep_count = min(
                args.num_eval_envs, args.num_eval_episodes - batch_start
            )
            for env_index in range(keep_count):
                rows.append(
                    {
                        "mode": mode,
                        "num_distractors": num_distractors,
                        "episode_index": batch_start + env_index,
                        "episode_seed": episode_seeds[env_index],
                        "success_once": bool(
                            tensor_item(episode_info["success_once"], env_index)
                        ),
                        "success_at_end": bool(
                            tensor_item(episode_info["success_at_end"], env_index)
                        ),
                        "return": float(
                            tensor_item(episode_info["return"], env_index)
                        ),
                        "episode_len": int(
                            tensor_item(episode_info["episode_len"], env_index)
                        ),
                    }
                )
    finally:
        env.close()

    return rows


def metric_summary(rows: list[dict]) -> dict:
    summary = {}
    for metric in ("success_once", "success_at_end", "return", "episode_len"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    summary["num_episodes"] = len(rows)
    return summary


def build_summary(args: Args, rows: list[dict]) -> dict:
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "args": asdict(args),
        "settings": {},
        "paired_vcd_minus_vanilla": {},
    }
    for count in args.distractor_counts:
        count_key = str(count)
        result["settings"][count_key] = {}
        by_mode = {}
        for mode in ("vanilla", "vcd"):
            setting_rows = [
                row
                for row in rows
                if row["mode"] == mode and row["num_distractors"] == count
            ]
            by_mode[mode] = {
                row["episode_seed"]: row for row in setting_rows
            }
            result["settings"][count_key][mode] = metric_summary(setting_rows)

        paired_summary = {}
        paired_seeds = sorted(set(by_mode["vanilla"]) & set(by_mode["vcd"]))
        for metric in ("success_once", "success_at_end", "return", "episode_len"):
            deltas = np.asarray(
                [
                    float(by_mode["vcd"][seed][metric])
                    - float(by_mode["vanilla"][seed][metric])
                    for seed in paired_seeds
                ],
                dtype=np.float64,
            )
            paired_summary[metric] = {
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std()),
            }
        paired_summary["num_paired_episodes"] = len(paired_seeds)
        result["paired_vcd_minus_vanilla"][count_key] = paired_summary
    return result


def write_results(output_dir: Path, args: Args, rows: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "episodes.csv"
    fieldnames = [
        "mode",
        "num_distractors",
        "episode_index",
        "episode_seed",
        "success_once",
        "success_at_end",
        "return",
        "episode_len",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(args, rows)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(f"Wrote per-episode results to {csv_path}")
    print(f"Wrote aggregate results to {summary_path}")
    for count in args.distractor_counts:
        count_summary = summary["settings"][str(count)]
        vanilla = count_summary["vanilla"]["success_once"]["mean"]
        vcd = count_summary["vcd"]["success_once"]["mean"]
        delta = summary["paired_vcd_minus_vanilla"][str(count)]["success_once"][
            "mean_delta"
        ]
        print(
            f"distractors={count}: vanilla_success={vanilla:.4f}, "
            f"vcd_success={vcd:.4f}, paired_delta={delta:+.4f}"
        )


def main():
    args = tyro.cli(Args)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.output_dir is None:
        output_dir = (
            Path(args.checkpoint).resolve().parent / f"vcd_eval_{int(time.time())}"
        )
    else:
        output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    checkpoint_state = torch.load(args.checkpoint, map_location="cpu")
    all_rows = []
    for count in args.distractor_counts:
        for mode in ("vanilla", "vcd"):
            print(f"Evaluating mode={mode}, num_distractors={count}")
            all_rows.extend(
                evaluate_setting(
                    args,
                    mode=mode,
                    num_distractors=count,
                    checkpoint_state=checkpoint_state,
                    output_dir=output_dir,
                )
            )
    write_results(output_dir, args, all_rows)


if __name__ == "__main__":
    main()
