from __future__ import annotations

import argparse
from collections import deque
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl.agents import DQNAgent, DQNConfig
from rl.environment import PacmanEnv
from rl.environment.rewards import RewardConfig
from rl.training.experiment_workflow import run_evaluation
from rl.training.metrics import EpisodeMetricsLogger, create_run_directory, software_versions, utc_now


def _run_configuration(args: argparse.Namespace, agent: DQNAgent, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "max_steps_per_episode": args.max_steps,
        "total_training_steps": args.total_steps if mode == "train" else None,
        "max_episodes": args.max_episodes if mode == "train" else args.eval_episodes,
        "checkpoint_every_steps": getattr(args, "checkpoint_every_steps", None),
        "evaluation_every_steps": getattr(args, "evaluation_every_steps", None),
        "checkpoint_evaluation_episodes": getattr(args, "checkpoint_evaluation_episodes", None),
        "dqn": asdict(agent.config),
        "resolved_device": str(agent.device),
        "observation_shape": agent.observation_shape,
        "action_count": agent.action_count,
        "bootstrap_on_truncation": True,
        "network_architecture": {
            "type": "CNN",
            "convolution_channels": list(agent.config.conv_channels),
            "hidden_size": agent.config.hidden_size,
        },
        "reward_configuration": asdict(RewardConfig()),
        "software_versions": software_versions(),
    }


def train(
    *,
    total_steps: int,
    max_steps: int,
    max_episodes: int | None = None,
    output_root: Path = Path("runs"),
    config: DQNConfig | None = None,
    resume_checkpoint: Path | None = None,
    checkpoint_every_steps: int | None = 2_500,
    evaluation_every_steps: int | None = 5_000,
    checkpoint_evaluation_episodes: int = 5,
    evaluation_seed: int | None = None,
) -> Path:
    if total_steps <= 0 or max_steps <= 0 or (max_episodes is not None and max_episodes <= 0):
        raise ValueError("Training limits must be positive.")
    if (
        (checkpoint_every_steps is not None and checkpoint_every_steps <= 0)
        or (evaluation_every_steps is not None and evaluation_every_steps <= 0)
        or checkpoint_evaluation_episodes <= 0
    ):
        raise ValueError("Checkpoint and evaluation intervals and episode counts must be positive.")

    env = PacmanEnv(max_steps=max_steps)
    agent: DQNAgent | None = None
    logger: EpisodeMetricsLogger | None = None
    run_directory: Path | None = None
    run_status = "completed"
    started_at = time.perf_counter()
    recent_episodes: deque[dict[str, float | int | bool | None]] = deque(maxlen=20)
    checkpoint_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    last_training_log_step = 0
    episode_return = 0.0
    episode_length = 0
    try:
        observation, reset_info = env.reset(seed=(config.seed if config else None))
        if resume_checkpoint is not None:
            agent = DQNAgent.load_checkpoint(
                resume_checkpoint,
                tuple(env.observation_space.shape),
                int(env.action_space.n),
                device=config.device if config else None,
            )
        else:
            agent = DQNAgent(
                tuple(env.observation_space.shape),
                int(env.action_space.n),
                config=config,
            )
        run_directory = create_run_directory(output_root, "pacman-dqn", agent.config.seed)
        experiment_id = run_directory.name
        logger = EpisodeMetricsLogger(run_directory, _run_configuration(
            argparse.Namespace(
                max_steps=max_steps,
                total_steps=total_steps,
                max_episodes=max_episodes,
                eval_episodes=None,
                checkpoint_every_steps=checkpoint_every_steps,
                evaluation_every_steps=evaluation_every_steps,
                checkpoint_evaluation_episodes=checkpoint_evaluation_episodes,
            ),
            agent,
            "train",
        ), metadata={
            "experiment_id": experiment_id,
            "started_at": utc_now(),
            "status": "running",
            "seed": agent.config.seed,
            "software_versions": software_versions(),
            "device": str(agent.device),
            "environment": {
                "base_url": env.game_client.base_url,
                "max_steps_per_episode": max_steps,
                "observation_shape": list(env.observation_space.shape),
                "action_space": str(env.action_space),
            },
        })

        next_checkpoint_step = (
            ((agent.global_step // checkpoint_every_steps) + 1) * checkpoint_every_steps
            if checkpoint_every_steps is not None
            else None
        )
        next_evaluation_step = (
            ((agent.global_step // evaluation_every_steps) + 1) * evaluation_every_steps
            if evaluation_every_steps is not None
            else None
        )
        initial_state = reset_info.get("state", {})
        episode_started = time.perf_counter()
        while agent.global_step < total_steps:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.replay_buffer.add(
                observation,
                action,
                float(reward),
                next_observation,
                terminated,
                truncated,
            )
            agent.global_step += 1
            episode_return += float(reward)
            episode_length += 1
            agent.learn()
            observation = next_observation

            if terminated or truncated:
                agent.completed_episodes += 1
                final_state = info.get("state", {})
                if not isinstance(final_state, dict):
                    final_state = {}
                if not isinstance(initial_state, dict):
                    initial_state = {}
                game_score = info.get("game_score", final_state.get("score"))
                pellets_remaining = final_state.get("pelletsRemaining")
                initial_pellets = initial_state.get("pelletsRemaining")
                initial_lives = initial_state.get("lives")
                final_lives = final_state.get("lives")
                pellets_collected = (
                    int(initial_pellets) - int(pellets_remaining)
                    if isinstance(initial_pellets, (int, float)) and isinstance(pellets_remaining, (int, float))
                    else None
                )
                lives_lost = (
                    int(initial_lives) - int(final_lives)
                    if isinstance(initial_lives, (int, float)) and isinstance(final_lives, (int, float))
                    else None
                )
                won = bool(final_state["won"]) if "won" in final_state else None
                termination_reason = (
                    "won" if won else "game_over" if terminated else "max_steps" if truncated else None
                )
                duration_seconds = time.perf_counter() - episode_started
                episode_metrics = {
                    "episode": agent.completed_episodes,
                    "global_env_step": agent.global_step,
                    "episode_return": episode_return,
                    "game_score": game_score,
                    "pellets_collected": pellets_collected,
                    "pellets_remaining": pellets_remaining,
                    "episode_length": episode_length,
                    "duration_seconds": duration_seconds,
                    "lives_lost": lives_lost,
                    "won": won,
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_reason": termination_reason,
                    "epsilon": agent.epsilon(),
                    "latest_loss": agent.latest_loss,
                    "gradient_updates": agent.gradient_updates,
                }
                logger.log_episode(
                    episode_metrics
                )
                recent_episodes.append(episode_metrics)
                elapsed = time.perf_counter() - started_at
                recent = list(recent_episodes)
                values = lambda key: [float(item[key]) for item in recent if isinstance(item.get(key), (int, float))]
                collected_values = values("pellets_collected")
                remaining_values = values("pellets_remaining")
                logger.log_training({
                    "global_env_step": agent.global_step,
                    "episodes_completed": agent.completed_episodes,
                    "window_episodes": len(recent),
                    "mean_episode_return": sum(values("episode_return")) / max(1, len(values("episode_return"))),
                    "mean_game_score": (
                        sum(values("game_score")) / len(values("game_score")) if values("game_score") else None
                    ),
                    "mean_pellets_collected": (
                        sum(collected_values) / len(collected_values) if collected_values else None
                    ),
                    "mean_pellets_remaining": (
                        sum(remaining_values) / len(remaining_values) if remaining_values else None
                    ),
                    "completion_rate": (
                        sum(bool(item["won"]) for item in recent if item["terminated"] and item["won"] is not None)
                        / max(1, sum(item["terminated"] and item["won"] is not None for item in recent))
                        if any(item["terminated"] and item["won"] is not None for item in recent)
                        else None
                    ),
                    "mean_episode_length": sum(values("episode_length")) / max(1, len(values("episode_length"))),
                    "latest_loss": agent.latest_loss,
                    "epsilon": agent.epsilon(),
                    "learning_rate": agent.optimizer.param_groups[0]["lr"],
                    "gradient_updates": agent.gradient_updates,
                    "steps_per_second": agent.global_step / elapsed if elapsed > 0 else None,
                    "elapsed_seconds": elapsed,
                })
                last_training_log_step = agent.global_step
                print(
                    f"episode={agent.completed_episodes} step={agent.global_step} "
                    f"return={episode_return:.2f} game_score={game_score} "
                    f"length={episode_length} terminated={terminated} truncated={truncated} "
                    f"epsilon={agent.epsilon():.3f} loss={agent.latest_loss}"
                )
                checkpoint_due = (
                    next_checkpoint_step is not None and agent.global_step >= next_checkpoint_step
                )
                evaluation_due = (
                    next_evaluation_step is not None and agent.global_step >= next_evaluation_step
                )
                if checkpoint_due or evaluation_due:
                    snapshot = run_directory / "checkpoints" / f"step-{agent.global_step:08d}.pt"
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    agent.save_checkpoint(snapshot)
                    checkpoint_history.append({
                        "global_env_step": agent.global_step,
                        "episodes_completed": agent.completed_episodes,
                        "gradient_updates": agent.gradient_updates,
                        "epsilon": agent.epsilon(),
                        "latest_loss": agent.latest_loss,
                        "checkpoint": str(snapshot.resolve()),
                    })
                if evaluation_due:
                    evaluation_directory = run_evaluation(
                        snapshot,
                        episodes=checkpoint_evaluation_episodes,
                        max_steps=max_steps,
                        output_root=run_directory / "evaluations",
                        seed=agent.config.seed if evaluation_seed is None else evaluation_seed,
                    )
                    evaluation_history.append({
                        **checkpoint_history[-1],
                        "evaluation_directory": str(evaluation_directory.resolve()),
                    })
                    print(
                        f"checkpoint_evaluation step={agent.global_step} "
                        f"episodes={checkpoint_evaluation_episodes} path={evaluation_directory}"
                    )
                if checkpoint_due:
                    while next_checkpoint_step is not None and next_checkpoint_step <= agent.global_step:
                        next_checkpoint_step += checkpoint_every_steps
                if evaluation_due:
                    while next_evaluation_step is not None and next_evaluation_step <= agent.global_step:
                        next_evaluation_step += evaluation_every_steps
                if max_episodes is not None and agent.completed_episodes >= max_episodes:
                    break
                observation, reset_info = env.reset()
                initial_state = reset_info.get("state", {})
                episode_return = 0.0
                episode_length = 0
                episode_started = time.perf_counter()

        if run_status == "completed" and agent.global_step > last_training_log_step:
            elapsed = time.perf_counter() - started_at
            logger.log_training({
                "global_env_step": agent.global_step,
                "episodes_completed": agent.completed_episodes,
                "window_episodes": 0,
                "latest_loss": agent.latest_loss,
                "epsilon": agent.epsilon(),
                "learning_rate": agent.optimizer.param_groups[0]["lr"],
                "gradient_updates": agent.gradient_updates,
                "steps_per_second": agent.global_step / elapsed if elapsed > 0 else None,
                "elapsed_seconds": elapsed,
            })

        checkpoint_due = (
            next_checkpoint_step is not None and agent.global_step >= next_checkpoint_step
        )
        evaluation_due = (
            run_status == "completed"
            and next_evaluation_step is not None
            and agent.global_step >= next_evaluation_step
        )
        if run_status == "completed" and (checkpoint_due or evaluation_due):
            snapshot = run_directory / "checkpoints" / f"step-{agent.global_step:08d}.pt"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            agent.save_checkpoint(snapshot)
            checkpoint_history.append({
                "global_env_step": agent.global_step,
                "episodes_completed": agent.completed_episodes,
                "gradient_updates": agent.gradient_updates,
                "epsilon": agent.epsilon(),
                "latest_loss": agent.latest_loss,
                "checkpoint": str(snapshot.resolve()),
            })
        if evaluation_due:
            evaluation_directory = run_evaluation(
                snapshot,
                episodes=checkpoint_evaluation_episodes,
                max_steps=max_steps,
                output_root=run_directory / "evaluations",
                seed=agent.config.seed if evaluation_seed is None else evaluation_seed,
            )
            evaluation_history.append({
                **checkpoint_history[-1],
                "evaluation_directory": str(evaluation_directory.resolve()),
            })
            print(
                f"checkpoint_evaluation step={agent.global_step} "
                f"episodes={checkpoint_evaluation_episodes} path={evaluation_directory}"
            )
        return run_directory
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("Training interrupted; saving the latest checkpoint.", file=sys.stderr)
        return run_directory or Path()
    except Exception:
        run_status = "failed"
        raise
    finally:
        try:
            if agent is not None and run_directory is not None:
                agent.save_checkpoint(run_directory / "checkpoint.pt")
        finally:
            try:
                if logger is not None:
                    logger.finish(
                        run_status,
                        global_env_step=agent.global_step if agent is not None else 0,
                        completed_episodes=agent.completed_episodes if agent is not None else 0,
                        gradient_updates=agent.gradient_updates if agent is not None else 0,
                        epsilon=agent.epsilon() if agent is not None else None,
                        latest_loss=agent.latest_loss if agent is not None else None,
                        partial_episode_steps=(
                            episode_length
                            if agent is not None and agent.global_step > last_training_log_step
                            else 0
                        ),
                        elapsed_seconds=time.perf_counter() - started_at,
                        checkpoints=checkpoint_history if agent is not None else [],
                        checkpoint_evaluations=evaluation_history if agent is not None else [],
                    )
            finally:
                if logger is not None:
                    logger.close()
                env.close()


def evaluate(
    checkpoint: Path,
    *,
    episodes: int,
    max_steps: int,
    output_root: Path,
    device: str | None = None,
    seed: int = 7,
) -> Path:
    return run_evaluation(
        checkpoint,
        episodes=episodes,
        max_steps=max_steps,
        output_root=output_root,
        device=device,
        seed=seed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate a CNN-based DQN on the live Pac-Man game.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="Resume training from a DQN checkpoint.")
    mode.add_argument("--checkpoint", type=Path, help="Evaluate a checkpoint greedily instead of training.")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=200, help="RL time-limit per episode.")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=2_500, help="Use 0 to disable stage checkpoints.")
    parser.add_argument("--evaluation-every-steps", type=int, default=5_000, help="Use 0 to disable checkpoint evaluations.")
    parser.add_argument("--checkpoint-evaluation-episodes", type=int, default=5)
    return parser


def _config_from_args(args: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_interval=args.target_update_interval,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        device=args.device or "auto",
    )


def main() -> None:
    args = _parser().parse_args()
    if args.checkpoint is not None:
        run_directory = evaluate(
            args.checkpoint,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            output_root=args.output_root,
            device=args.device,
            seed=args.seed,
        )
    else:
        config = None
        if args.resume is None:
            config = _config_from_args(args)
        run_directory = train(
            total_steps=args.total_steps,
            max_steps=args.max_steps,
            max_episodes=args.max_episodes,
            output_root=args.output_root,
            config=config,
            resume_checkpoint=args.resume,
            checkpoint_every_steps=args.checkpoint_every_steps or None,
            evaluation_every_steps=args.evaluation_every_steps or None,
            checkpoint_evaluation_episodes=args.checkpoint_evaluation_episodes,
            evaluation_seed=args.evaluation_seed,
        )
    print(f"Run artifacts: {run_directory}")


if __name__ == "__main__":
    main()
