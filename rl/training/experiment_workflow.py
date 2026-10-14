from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl.agents import DQNAgent
from rl.environment import PacmanEnv
from rl.environment.rewards import RewardConfig
from rl.training.metrics import EpisodeMetricsLogger, create_run_directory, software_versions, utc_now


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def numbers(field: str) -> list[float]:
        return [number for row in rows if (number := _numeric(row.get(field))) is not None]

    def aggregate(field: str) -> dict[str, float | int | None]:
        values = numbers(field)
        return {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }

    known_results = [
        str(row.get("won", "")).lower() in {"true", "1"}
        for row in rows
        if str(row.get("truncated", "")).lower() not in {"true", "1"}
        and str(row.get("won", "")).lower() in {"true", "false", "0", "1"}
    ]
    episode_lengths = numbers("episode_length")
    return {
        "episodes": len(rows),
        "environment_steps": (
            int(sum(episode_lengths))
            if rows and len(episode_lengths) == len(rows)
            else None
        ),
        "episode_return": aggregate("episode_return"),
        "game_score": aggregate("game_score"),
        "pellets_collected": aggregate("pellets_collected"),
        "pellets_remaining": aggregate("pellets_remaining"),
        "episode_length": aggregate("episode_length"),
        "duration_seconds": aggregate("duration_seconds"),
        "lives_lost": aggregate("lives_lost"),
        "completion_rate": float(sum(known_results) / len(known_results)) if known_results else None,
        "wins_observed": sum(known_results),
        "completion_observations": len(known_results),
        "truncated_episodes": sum(str(row.get("truncated", "")).lower() == "true" for row in rows),
        "terminated_episodes": sum(str(row.get("terminated", "")).lower() == "true" for row in rows),
    }


def _episode_summary(directory: Path) -> dict[str, Any]:
    summary = _read_json(directory / "summary.json")
    rows = _read_csv(directory / "episodes.csv")
    if rows:
        summary.update(summarize_episodes(rows))
    return summary


def _state_value(info: Mapping[str, Any], key: str) -> Any:
    state = info.get("state")
    if isinstance(state, Mapping):
        return state.get(key)
    return None


def _episode_metrics(
    *,
    episode: int,
    global_step: int,
    episode_return: float,
    episode_length: int,
    duration_seconds: float,
    initial_info: Mapping[str, Any],
    final_info: Mapping[str, Any],
    terminated: bool,
    truncated: bool,
    epsilon: float | None,
    latest_loss: float | None,
    gradient_updates: int,
) -> dict[str, Any]:
    initial_pellets = _state_value(initial_info, "pelletsRemaining")
    pellets_remaining = _state_value(final_info, "pelletsRemaining")
    initial_lives = _state_value(initial_info, "lives")
    final_lives = _state_value(final_info, "lives")
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
    won = _state_value(final_info, "won")
    if not isinstance(won, bool):
        won = None
    game_score = final_info.get("game_score", _state_value(final_info, "score"))
    if not isinstance(game_score, (int, float)):
        game_score = None
    reason = "won" if won is True else "game_over" if terminated else "max_steps" if truncated else None
    return {
        "episode": episode,
        "global_env_step": global_step,
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
        "termination_reason": reason,
        "epsilon": epsilon,
        "latest_loss": latest_loss,
        "gradient_updates": gradient_updates,
    }


def _experiment_configuration(
    env: PacmanEnv,
    *,
    mode: str,
    episodes: int,
    max_steps: int,
    seed: int,
    policy: str,
    agent: DQNAgent | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "mode": mode,
        "policy": policy,
        "seed": seed,
        "evaluation_episodes": episodes,
        "max_steps_per_episode": max_steps,
        "observation_shape": list(env.observation_space.shape),
        "action_count": int(env.action_space.n),
        "reward_configuration": vars(RewardConfig()),
        "software_versions": software_versions(),
        "environment": {
            "base_url": env.game_client.base_url,
            "max_steps_per_episode": max_steps,
        },
        "policy_updates_enabled": False,
    }
    if agent is not None:
        configuration["dqn"] = {
            "learning_rate": agent.config.learning_rate,
            "gamma": agent.config.gamma,
            "batch_size": agent.config.batch_size,
            "replay_capacity": agent.config.replay_capacity,
            "warmup_steps": agent.config.warmup_steps,
            "epsilon_start": agent.config.epsilon_start,
            "epsilon_end": agent.config.epsilon_end,
            "epsilon_decay_steps": agent.config.epsilon_decay_steps,
            "target_update_interval": agent.config.target_update_interval,
            "gradient_clip": agent.config.gradient_clip,
            "hidden_size": agent.config.hidden_size,
            "conv_channels": list(agent.config.conv_channels),
        }
        configuration["network_architecture"] = {
            "type": "CNN",
            "convolution_channels": list(agent.config.conv_channels),
            "hidden_size": agent.config.hidden_size,
        }
        configuration["resolved_device"] = str(agent.device)
        configuration["training_checkpoint"] = {
            "global_env_step": agent.global_step,
            "episodes_completed": agent.completed_episodes,
            "gradient_updates": agent.gradient_updates,
            "epsilon": agent.epsilon(),
            "latest_loss": agent.latest_loss,
        }
    if checkpoint is not None:
        configuration["checkpoint"] = str(checkpoint.resolve())
    return configuration


def _execute_episodes(
    *,
    env: PacmanEnv,
    logger: EpisodeMetricsLogger,
    episodes: int,
    action_for_observation: Callable[[np.ndarray], int],
    seed: int,
) -> list[dict[str, Any]]:
    observation, initial_info = env.reset(seed=seed)
    result_rows: list[dict[str, Any]] = []
    global_step = 0
    for episode in range(1, episodes + 1):
        episode_started = time.perf_counter()
        episode_return = 0.0
        episode_length = 0
        while True:
            action = action_for_observation(observation)
            observation, reward, terminated, truncated, final_info = env.step(action)
            global_step += 1
            episode_length += 1
            episode_return += float(reward)
            if terminated or truncated:
                row = _episode_metrics(
                    episode=episode,
                    global_step=global_step,
                    episode_return=episode_return,
                    episode_length=episode_length,
                    duration_seconds=time.perf_counter() - episode_started,
                    initial_info=initial_info,
                    final_info=final_info,
                    terminated=terminated,
                    truncated=truncated,
                    epsilon=0.0,
                    latest_loss=None,
                    gradient_updates=0,
                )
                logger.log_episode(row)
                result_rows.append(row)
                break
        if episode < episodes:
            observation, initial_info = env.reset()
    return result_rows


def _write_summary(run_directory: Path, rows: Sequence[Mapping[str, Any]], status: str) -> dict[str, Any]:
    summary = {
        "status": status,
        "generated_at": utc_now(),
        **summarize_episodes(rows),
    }
    with (run_directory / "summary.json").open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    return summary


def run_evaluation(
    checkpoint: str | Path,
    *,
    episodes: int,
    max_steps: int,
    output_root: str | Path = Path("evaluations"),
    seed: int = 7,
    device: str | None = None,
) -> Path:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("Evaluation episode and step limits must be positive.")
    checkpoint_path = Path(checkpoint)
    env = PacmanEnv(max_steps=max_steps)
    logger: EpisodeMetricsLogger | None = None
    run_directory: Path | None = None
    rows: list[dict[str, Any]] = []
    status = "completed"
    try:
        agent = DQNAgent.load_checkpoint(
            checkpoint_path,
            tuple(env.observation_space.shape),
            int(env.action_space.n),
            device=device,
        )
        agent.online_network.eval()
        agent.target_network.eval()
        run_directory = create_run_directory(output_root, "pacman-evaluation", seed)
        configuration = _experiment_configuration(
            env,
            mode="evaluation",
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            policy="greedy",
            agent=agent,
            checkpoint=checkpoint_path,
        )
        logger = EpisodeMetricsLogger(run_directory, configuration, {
            "experiment_id": run_directory.name,
            "started_at": utc_now(),
            "status": "running",
            "software_versions": software_versions(),
            "device": str(agent.device),
            "evaluation_seed": seed,
            "policy_updates_enabled": False,
        })
        rows = _execute_episodes(
            env=env,
            logger=logger,
            episodes=episodes,
            action_for_observation=lambda observation: agent.select_action(observation, explore=False),
            seed=seed,
        )
        _write_summary(run_directory, rows, status)
        logger.finish(status, completed_episodes=len(rows), policy_updates_enabled=False)
        return run_directory
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        if logger is not None:
            try:
                if run_directory is not None and not (run_directory / "summary.json").exists():
                    rows = _read_csv(run_directory / "episodes.csv")
                    _write_summary(run_directory, rows, status)
                if run_directory is not None and (run_directory / "run_metadata.json").exists():
                    metadata = _read_json(run_directory / "run_metadata.json")
                    if metadata.get("status") == "running":
                        logger.finish(status, completed_episodes=len(rows), policy_updates_enabled=False)
            finally:
                logger.close()
        env.close()


def run_random_baseline(
    *,
    episodes: int,
    max_steps: int,
    output_root: str | Path = Path("evaluations"),
    seed: int = 7,
) -> Path:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("Baseline episode and step limits must be positive.")
    env = PacmanEnv(max_steps=max_steps)
    logger: EpisodeMetricsLogger | None = None
    run_directory: Path | None = None
    rows: list[dict[str, Any]] = []
    status = "completed"
    try:
        env.action_space.seed(seed)
        run_directory = create_run_directory(output_root, "pacman-random-baseline", seed)
        configuration = _experiment_configuration(
            env,
            mode="evaluation",
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            policy="uniform_random_valid_action",
        )
        logger = EpisodeMetricsLogger(run_directory, configuration, {
            "experiment_id": run_directory.name,
            "started_at": utc_now(),
            "status": "running",
            "software_versions": software_versions(),
            "device": "cpu",
            "evaluation_seed": seed,
            "policy_updates_enabled": False,
        })

        def sample_valid_action(observation: np.ndarray) -> int:
            _ = observation
            return int(env.action_space.sample())

        rows = _execute_episodes(
            env=env,
            logger=logger,
            episodes=episodes,
            action_for_observation=sample_valid_action,
            seed=seed,
        )
        _write_summary(run_directory, rows, status)
        logger.finish(status, completed_episodes=len(rows), policy_updates_enabled=False)
        return run_directory
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        if logger is not None:
            try:
                if run_directory is not None and not (run_directory / "summary.json").exists():
                    rows = _read_csv(run_directory / "episodes.csv")
                    _write_summary(run_directory, rows, status)
                if run_directory is not None and (run_directory / "run_metadata.json").exists():
                    metadata = _read_json(run_directory / "run_metadata.json")
                    if metadata.get("status") == "running":
                        logger.finish(status, completed_episodes=len(rows), policy_updates_enabled=False)
            finally:
                logger.close()
        env.close()


def compare_evaluations(
    dqn_directory: str | Path,
    baseline_directory: str | Path,
    *,
    output_root: str | Path = Path("comparisons"),
) -> Path:
    dqn_path = Path(dqn_directory)
    baseline_path = Path(baseline_directory)
    dqn_config = _read_json(dqn_path / "config.json")
    baseline_config = _read_json(baseline_path / "config.json")
    dqn_rows = _read_csv(dqn_path / "episodes.csv")
    baseline_rows = _read_csv(baseline_path / "episodes.csv")
    if not dqn_rows or not baseline_rows:
        raise ValueError("Both input experiment directories must contain non-empty episodes.csv files.")

    conditions = ("max_steps_per_episode", "reward_configuration", "observation_shape", "action_count", "seed")
    condition_differences = {
        key: {"dqn": dqn_config.get(key), "random": baseline_config.get(key)}
        for key in conditions
        if dqn_config.get(key) != baseline_config.get(key)
    }
    seed_differences = dqn_config.get("seed") != baseline_config.get("seed")
    seed_note = "The evaluation seeds differ." if seed_differences else "The evaluation seeds match."
    directory = create_run_directory(output_root, "pacman-comparison", 0)
    comparison = {
        "generated_at": utc_now(),
        "dqn_directory": str(dqn_path.resolve()),
        "random_baseline_directory": str(baseline_path.resolve()),
        "conditions_match": not condition_differences,
        "condition_differences": condition_differences,
        "seed_note": seed_note,
        "dqn": summarize_episodes(dqn_rows),
        "random_baseline": summarize_episodes(baseline_rows),
        "performance_claim": "No comparative performance claim is inferred from these results.",
    }
    with (directory / "comparison.json").open("x", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, sort_keys=True)
        file.write("\n")
    with (directory / "comparison.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("policy", "episode", "episode_return", "game_score", "pellets_collected", "pellets_remaining", "episode_length", "won", "terminated", "truncated"))
        writer.writeheader()
        for policy_name, policy_rows in (("dqn", dqn_rows), ("random_baseline", baseline_rows)):
            for row in policy_rows:
                writer.writerow({"policy": policy_name, **{field: row.get(field) for field in writer.fieldnames if field != "policy"}})
    return directory


def _plot_series(axis: Any, rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str, label: str) -> bool:
    points = [
        (_numeric(row.get(x_key)), _numeric(row.get(y_key)))
        for row in rows
    ]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        axis.text(0.5, 0.5, "No recorded data", ha="center", va="center", transform=axis.transAxes)
        return False
    x_values, y_values = zip(*points)
    axis.plot(x_values, y_values, marker="o", markersize=3, label=label)
    return True


def _save_plot(
    output_directory: Path,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[Sequence[Mapping[str, Any]], str, str, str]],
) -> list[str]:
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    any_data = False
    for rows, x_key, y_key, label in series:
        any_data = _plot_series(axis, rows, x_key, y_key, label) or any_data
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    if any_data and len(series) > 1:
        axis.legend()
    output_directory.mkdir(parents=True, exist_ok=True)
    png = output_directory / f"{filename}.png"
    pdf = output_directory / f"{filename}.pdf"
    figure.savefig(png, dpi=150)
    figure.savefig(pdf)
    plt.close(figure)
    return [png.name, pdf.name]


def generate_plots(
    experiment_directory: str | Path,
    *,
    evaluation_directory: str | Path | None = None,
    baseline_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    smoothing_window: int = 10,
) -> Path:
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive.")
    experiment_path = Path(experiment_directory)
    if output_directory is None:
        output_directory = create_run_directory(experiment_path, "plots", 0)
    plots_path = Path(output_directory)
    if plots_path.exists() and any(plots_path.iterdir()):
        raise FileExistsError(f"Plot output directory is not empty: {plots_path}")
    plots_path.mkdir(parents=True, exist_ok=True)
    training_rows = _read_csv(experiment_path / "training.csv")
    episode_rows = _read_csv(experiment_path / "episodes.csv")
    eval_rows = _read_csv(Path(evaluation_directory) / "episodes.csv") if evaluation_directory else []
    random_rows = _read_csv(Path(baseline_directory) / "episodes.csv") if baseline_directory else []
    checkpoint_evaluation_rows: list[dict[str, Any]] = []
    for summary_path in sorted((experiment_path / "evaluations").glob("*/summary.json")):
        stage_directory = summary_path.parent
        stage_config = _read_json(stage_directory / "config.json")
        stage_summary = _episode_summary(stage_directory)
        checkpoint_state = stage_config.get("training_checkpoint")
        if not isinstance(checkpoint_state, Mapping):
            continue
        checkpoint_evaluation_rows.append({
            "global_env_step": checkpoint_state.get("global_env_step"),
            "mean_return": stage_summary.get("episode_return", {}).get("mean"),
            "mean_game_score": stage_summary.get("game_score", {}).get("mean"),
        })
    files: list[str] = []
    files += _save_plot(plots_path, "episode_return", "Episode return", "Episode", "Episode return (sum of step rewards)", [(episode_rows, "episode", "episode_return", "DQN")])

    returns = [_numeric(row.get("episode_return")) for row in episode_rows]
    smooth_rows = [
        {"episode": row.get("episode"), "smoothed_return": float(np.mean([value for value in returns[max(0, i - smoothing_window + 1):i + 1] if value is not None])) if any(value is not None for value in returns[max(0, i - smoothing_window + 1):i + 1]) else None}
        for i, row in enumerate(episode_rows)
    ]
    files += _save_plot(plots_path, "smoothed_episode_return", f"Smoothed episode return (rolling window={smoothing_window} episodes)", "Episode", "Mean episode return (rolling)", [(smooth_rows, "episode", "smoothed_return", "DQN")])
    files += _save_plot(plots_path, "game_score", "Actual Pac-Man game score", "Episode", "Game score (server state.score)", [(episode_rows, "episode", "game_score", "DQN")])
    files += _save_plot(plots_path, "pellets", "Pellet collection", "Episode", "Pellets", [(episode_rows, "episode", "pellets_collected", "Collected"), (episode_rows, "episode", "pellets_remaining", "Remaining")])
    files += _save_plot(plots_path, "episode_length", "Episode length", "Episode", "Environment steps", [(episode_rows, "episode", "episode_length", "DQN")])
    files += _save_plot(plots_path, "training_loss", "DQN training loss", "Global environment step", "Latest episode loss", [(training_rows, "global_env_step", "latest_loss", "Loss")])
    files += _save_plot(plots_path, "epsilon", "Epsilon schedule", "Global environment step", "Epsilon", [(training_rows, "global_env_step", "epsilon", "Epsilon")])
    files += _save_plot(plots_path, "gradient_updates", "Gradient updates", "Global environment step", "Gradient update count", [(training_rows, "global_env_step", "gradient_updates", "Updates")])
    files += _save_plot(plots_path, "evaluation_return", "Evaluation episode return", "Episode", "Episode return (sum of step rewards)", [(eval_rows, "episode", "episode_return", "Greedy DQN")])
    files += _save_plot(plots_path, "evaluation_game_score", "Evaluation actual game score", "Episode", "Game score (server state.score)", [(eval_rows, "episode", "game_score", "Greedy DQN")])
    files += _save_plot(plots_path, "checkpoint_evaluation_return", "Checkpoint evaluation return", "Training environment steps", "Mean evaluation return", [(checkpoint_evaluation_rows, "global_env_step", "mean_return", "Greedy checkpoint")])
    files += _save_plot(plots_path, "checkpoint_evaluation_game_score", "Checkpoint evaluation game score", "Training environment steps", "Mean evaluation game score", [(checkpoint_evaluation_rows, "global_env_step", "mean_game_score", "Greedy checkpoint")])
    files += _save_plot(plots_path, "dqn_vs_random", "Greedy DQN vs random baseline", "Episode", "Metric value", [(eval_rows, "episode", "episode_return", "DQN return"), (random_rows, "episode", "episode_return", "Random return")])
    files += _save_plot(plots_path, "dqn_vs_random_game_score", "Greedy DQN vs random baseline: game score", "Episode", "Game score (server state.score)", [(eval_rows, "episode", "game_score", "DQN game score"), (random_rows, "episode", "game_score", "Random game score")])
    manifest = {
        "generated_at": utc_now(),
        "experiment_directory": str(experiment_path.resolve()),
        "evaluation_directory": str(Path(evaluation_directory).resolve()) if evaluation_directory else None,
        "baseline_directory": str(Path(baseline_directory).resolve()) if baseline_directory else None,
        "smoothing_window_episodes": smoothing_window,
        "files": files,
        "pdf_generated": True,
    }
    (plots_path / "plots.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return plots_path


def generate_report(
    experiment_directory: str | Path,
    *,
    evaluation_directory: str | Path | None = None,
    baseline_directory: str | Path | None = None,
    output_root: str | Path = Path("reports"),
    smoothing_window: int = 10,
) -> Path:
    experiment_path = Path(experiment_directory)
    config = _read_json(experiment_path / "config.json")
    metadata = _read_json(experiment_path / "run_metadata.json")
    training_rows = _read_csv(experiment_path / "episodes.csv")
    training_summary = summarize_episodes(training_rows)
    last_training_row = training_rows[-1] if training_rows else {}
    training_steps = metadata.get("global_env_step", _numeric(last_training_row.get("global_env_step")))
    gradient_updates = metadata.get("gradient_updates", _numeric(last_training_row.get("gradient_updates")))
    training_duration = metadata.get("elapsed_seconds")
    training_duration_display = (
        f"{float(training_duration):.2f} seconds"
        if isinstance(training_duration, (int, float))
        else "unavailable in this legacy experiment"
    )
    software = metadata.get("software_versions", config.get("software_versions"))
    if not software:
        software = "unavailable in this legacy experiment"
    reward_configuration = config.get("reward_configuration")
    if not reward_configuration:
        reward_configuration = "unavailable in this legacy experiment"
    evaluation_path = Path(evaluation_directory) if evaluation_directory else None
    baseline_path = Path(baseline_directory) if baseline_directory else None
    eval_summary = _episode_summary(evaluation_path) if evaluation_path else {}
    baseline_summary = _episode_summary(baseline_path) if baseline_path else {}
    checkpoint_evaluations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for summary_path in sorted((experiment_path / "evaluations").glob("*/summary.json")):
        stage_directory = summary_path.parent
        stage_config = _read_json(stage_directory / "config.json")
        stage_summary = _episode_summary(stage_directory)
        checkpoint_state = stage_config.get("training_checkpoint")
        if isinstance(checkpoint_state, dict):
            checkpoint_evaluations.append((checkpoint_state, stage_summary))

    report_directory = create_run_directory(output_root, "pacman-report", int(config.get("dqn", {}).get("seed", 0)))
    plots_directory = report_directory / "plots"
    generate_plots(
        experiment_path,
        evaluation_directory=evaluation_path,
        baseline_directory=baseline_path,
        output_directory=plots_directory,
        smoothing_window=smoothing_window,
    )
    preliminary = len(training_rows) < 30 or not evaluation_path or not eval_summary
    assessment = "Insufficient evidence to assess learning from the available evaluations."
    dqn_config = _read_json(evaluation_path / "config.json") if evaluation_path else {}
    baseline_config = _read_json(baseline_path / "config.json") if baseline_path else {}
    condition_keys = ("max_steps_per_episode", "reward_configuration", "observation_shape", "action_count", "seed")
    condition_differences = [
        key for key in condition_keys if dqn_config.get(key) != baseline_config.get(key)
    ]
    if evaluation_path and baseline_path and eval_summary and baseline_summary:
        dqn_return = eval_summary.get("episode_return", {}).get("mean")
        random_return = baseline_summary.get("episode_return", {}).get("mean")
        dqn_score = eval_summary.get("game_score", {}).get("mean")
        random_score = baseline_summary.get("game_score", {}).get("mean")
        if condition_differences:
            assessment = f"Evaluation conditions differ ({', '.join(condition_differences)}); no cross-policy learning conclusion is supported."
        elif all(isinstance(value, (int, float)) for value in (dqn_return, random_return, dqn_score, random_score)):
            if dqn_return > random_return and dqn_score > random_score:
                assessment = "Measured results provide preliminary evidence of improvement over the matched random baseline; this single-seed evaluation does not establish robust learning."
            elif dqn_return <= random_return and dqn_score <= random_score:
                assessment = "The measured results provide no evidence of improvement over the matched random baseline."
            else:
                assessment = "Evaluation metrics are mixed; there is insufficient evidence to conclude that gameplay improved."
    report: list[str] = [
        "# Pac-Man DQN Experiment Report",
        "",
        f"**Assessment:** {'Preliminary; available evidence is too limited for a learning conclusion.' if preliminary else assessment}",
        "",
        "## Experiment overview",
        "",
        f"- Experiment ID: `{metadata.get('experiment_id', experiment_path.name)}`",
        f"- Training artifact directory: `{experiment_path.resolve()}`",
        f"- Run status: `{metadata.get('status', 'unknown')}`",
        f"- Started: `{metadata.get('started_at', 'unavailable')}`",
        f"- Finished: `{metadata.get('finished_at', 'unavailable')}`",
        f"- Software versions: `{json.dumps(software, sort_keys=True) if isinstance(software, Mapping) else software}`",
        f"- Device: `{metadata.get('device', config.get('resolved_device', 'unavailable'))}`",
        "",
        "## Environment and agent",
        "",
        f"- Observation shape: `{config.get('observation_shape', 'unavailable')}`",
        f"- Discrete action count: `{config.get('action_count', 'unavailable')}`",
        f"- Episode step limit: `{config.get('max_steps_per_episode', 'unavailable')}`",
        f"- DQN architecture: `{json.dumps(config.get('network_architecture', {'type': 'CNN', 'convolution_channels': config.get('dqn', {}).get('conv_channels'), 'hidden_size': config.get('dqn', {}).get('hidden_size')}), sort_keys=True)}`",
        f"- DQN hyperparameters: `{json.dumps(config.get('dqn', {}), sort_keys=True)}`",
        f"- Reward configuration: `{json.dumps(reward_configuration, sort_keys=True) if isinstance(reward_configuration, Mapping) else reward_configuration}`",
        f"- Time-limit policy: bootstrap from the valid final observation at truncation; true termination does not bootstrap.",
        "",
        "## Training results",
        "",
        f"- Recorded episodes: {training_summary['episodes']}",
        f"- Global steps: {training_steps if training_steps is not None else config.get('total_training_steps', 'unavailable')}",
        f"- Gradient updates: {gradient_updates if gradient_updates is not None else 'unavailable'}",
        f"- Recorded training duration: {training_duration_display}",
        f"- Checkpoint cadence: `{config.get('checkpoint_every_steps', 'unavailable')}` steps; checkpoint evaluation cadence: `{config.get('evaluation_every_steps', 'unavailable')}` steps using `{config.get('checkpoint_evaluation_episodes', 'unavailable')}` episodes.",
        f"- Replay warm-up: updates require both `{config.get('dqn', {}).get('warmup_steps', 'unavailable')}` global environment steps and `{config.get('dqn', {}).get('batch_size', 'unavailable')}` replay transitions.",
        f"- Epsilon progression: `{training_rows[0].get('epsilon') if training_rows else 'unavailable'}` to `{metadata.get('epsilon', checkpoint_evaluations[-1][0].get('epsilon') if checkpoint_evaluations else training_rows[-1].get('epsilon') if training_rows else 'unavailable')}`.",
        f"- Latest training loss: `{metadata.get('latest_loss', checkpoint_evaluations[-1][0].get('latest_loss') if checkpoint_evaluations else training_rows[-1].get('latest_loss') if training_rows else 'unavailable')}`.",
        f"- Final partial episode steps: `{metadata.get('partial_episode_steps', max(0, int(training_steps) - int(_numeric(last_training_row.get('global_env_step')) or 0)) if training_steps is not None else 'unavailable')}`.",
        f"- Mean episode return (sum of returned step rewards): {_format_aggregate(training_summary['episode_return'])}",
        f"- Actual game score mean: {_format_aggregate(training_summary['game_score'])}",
        f"- Mean pellets collected: {_format_aggregate(training_summary['pellets_collected'])}",
        f"- Maze completion rate: {_format_rate(training_summary['completion_rate'], training_summary['completion_observations'])}",
        f"- Mean episode length: {_format_aggregate(training_summary['episode_length'])}",
        "",
        "Return is computed from environment step rewards. Game score is independently read from the JavaScript state and is not substituted for return.",
        "",
        "## Evaluation and random baseline",
        "",
    ]
    if checkpoint_evaluations:
        report.extend([
            "### Checkpoint progress evaluations",
            "",
            "| Training steps | Episodes | Gradient updates | Epsilon | Training loss | Evaluation episodes | Evaluation steps | Mean return | Mean game score | Mean pellets | Mean episode length |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for checkpoint_state, stage_summary in checkpoint_evaluations:
            report.append(
                f"| {checkpoint_state.get('global_env_step', 'n/a')} "
                f"| {checkpoint_state.get('episodes_completed', 'n/a')} "
                f"| {checkpoint_state.get('gradient_updates', 'n/a')} "
                f"| {checkpoint_state.get('epsilon', 'n/a')} "
                f"| {checkpoint_state.get('latest_loss', 'n/a')} "
                f"| {stage_summary.get('episodes', 0)} "
                f"| {stage_summary.get('environment_steps', 'n/a')} "
                f"| {_format_aggregate(stage_summary.get('episode_return', {}))} "
                f"| {_format_aggregate(stage_summary.get('game_score', {}))} "
                f"| {_format_aggregate(stage_summary.get('pellets_collected', {}))} "
                f"| {_format_aggregate(stage_summary.get('episode_length', {}))} |"
            )
        report.append("")
    if evaluation_path and eval_summary:
        report.extend([
            f"- Greedy checkpoint evaluation: `{evaluation_path.resolve()}`",
            "- Evaluation policy: greedy action selection; epsilon exploration and model/optimizer updates are disabled.",
            f"- Episodes: {eval_summary.get('episodes', 0)}",
            f"- Environment steps: {eval_summary.get('environment_steps', 'unavailable')}",
            f"- Mean evaluation return: {_format_aggregate(eval_summary.get('episode_return', {}))}",
            f"- Mean evaluation game score: {_format_aggregate(eval_summary.get('game_score', {}))}",
            f"- Mean pellets collected: {_format_aggregate(eval_summary.get('pellets_collected', {}))}",
            f"- Mean episode length: {_format_aggregate(eval_summary.get('episode_length', {}))}",
            f"- Completion rate: {_format_rate(eval_summary.get('completion_rate'), eval_summary.get('completion_observations', 0))}",
        ])
    else:
        report.append("- No evaluation summary was supplied; evaluation conclusions are unavailable.")
    if baseline_path and baseline_summary:
        report.extend([
            f"- Random-action baseline: `{baseline_path.resolve()}`",
            f"- Baseline episodes: {baseline_summary.get('episodes', 0)}",
            f"- Baseline environment steps: {baseline_summary.get('environment_steps', 'unavailable')}",
            f"- Mean baseline return: {_format_aggregate(baseline_summary.get('episode_return', {}))}",
            f"- Mean baseline game score: {_format_aggregate(baseline_summary.get('game_score', {}))}",
            f"- Mean baseline pellets collected: {_format_aggregate(baseline_summary.get('pellets_collected', {}))}",
            f"- Mean baseline episode length: {_format_aggregate(baseline_summary.get('episode_length', {}))}",
            f"- Baseline completion rate: {_format_rate(baseline_summary.get('completion_rate'), baseline_summary.get('completion_observations', 0))}",
        ])
    else:
        report.append("- No random baseline summary was supplied; baseline comparison is unavailable.")

    if evaluation_path and baseline_path and eval_summary and baseline_summary:
        report.append(
            "- Evaluation conditions: "
            + (
                "matched for seed, episode limit, reward configuration, observation shape, and action count."
                if not condition_differences
                else f"mismatch in {', '.join(condition_differences)}."
            )
        )
        report.append(f"- Assessment: {assessment}")
        if eval_summary.get("completion_rate") is None:
            report.append("- DQN maze-completion rate is unavailable because all its evaluation episodes were truncated.")
        if baseline_summary.get("completion_rate") is None:
            report.append("- Random-baseline maze-completion rate is unavailable because all its evaluation episodes were truncated.")

    plot_manifest = _read_json(plots_directory / "plots.json")
    report.extend(["", "## Plots", "", f"Smoothing uses a rolling window of {smoothing_window} episodes; missing series remain empty rather than being imputed.", ""])
    for filename in plot_manifest.get("files", []):
        report.append(f"- [{filename}](plots/{filename})")

    report.extend([
        "",
        "## Limitations and known issues",
        "",
        "- This is one fixed-seed training run and one evaluation seed; repeated seeds are needed to establish robustness.",
        "- Game score and RL return are different metrics; inspect their separate columns.",
        "- The greedy DQN evaluation was time-limit truncated in every episode, so no maze completion was observed.",
        "- The project uses its local agent-control game model and approximate ghost behavior, not a cycle-accurate arcade implementation.",
        "",
        "## Suggested next experiment",
        "",
        "Repeat the training and matched evaluation with additional training/evaluation seeds, then aggregate per-seed metrics and completion outcomes before making a general performance claim.",
        "",
    ])
    report_path = report_directory / "report.md"
    with report_path.open("x", encoding="utf-8") as report_file:
        report_file.write("\n".join(report))
    report_metadata = {
        "generated_at": utc_now(),
        "training_directory": str(experiment_path.resolve()),
        "evaluation_directory": str(evaluation_path.resolve()) if evaluation_path else None,
        "baseline_directory": str(baseline_path.resolve()) if baseline_path else None,
        "preliminary": preliminary,
        "report": report_path.name,
        "plots_directory": "plots",
    }
    (report_directory / "report_metadata.json").write_text(json.dumps(report_metadata, indent=2) + "\n", encoding="utf-8")
    return report_directory


def _format_aggregate(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping) or value.get("mean") is None:
        return "unavailable"
    mean = float(value["mean"])
    std = value.get("std")
    return f"{mean:.3f} ± {float(std):.3f}" if std is not None else f"{mean:.3f}"


def _format_rate(rate: Any, count: Any) -> str:
    return f"{float(rate):.1%} (known for {count} episode(s))" if isinstance(rate, (int, float)) else "unavailable"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate, compare, plot, and report Pac-Man DQN experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Greedy evaluation of a saved DQN checkpoint.")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=5)
    evaluate_parser.add_argument("--max-steps", type=int, default=200)
    evaluate_parser.add_argument("--seed", type=int, default=7)
    evaluate_parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    evaluate_parser.add_argument("--output-root", type=Path, default=Path("evaluations"))

    random_parser = subparsers.add_parser("random", help="Evaluate the seeded uniform-random policy.")
    random_parser.add_argument("--episodes", type=int, default=5)
    random_parser.add_argument("--max-steps", type=int, default=200)
    random_parser.add_argument("--seed", type=int, default=7)
    random_parser.add_argument("--output-root", type=Path, default=Path("evaluations"))

    compare_parser = subparsers.add_parser("compare", help="Compare saved DQN and random evaluation results.")
    compare_parser.add_argument("--dqn", type=Path, required=True)
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--output-root", type=Path, default=Path("comparisons"))

    plot_parser = subparsers.add_parser("plot", help="Generate PNG and PDF plots from saved experiment CSV files.")
    plot_parser.add_argument("--experiment", type=Path, required=True)
    plot_parser.add_argument("--evaluation", type=Path)
    plot_parser.add_argument("--baseline", type=Path)
    plot_parser.add_argument("--output-directory", type=Path)
    plot_parser.add_argument("--smoothing-window", type=int, default=10)

    report_parser = subparsers.add_parser("report", help="Generate plots and a Markdown report from saved artifacts.")
    report_parser.add_argument("--experiment", type=Path, required=True)
    report_parser.add_argument("--evaluation", type=Path)
    report_parser.add_argument("--baseline", type=Path)
    report_parser.add_argument("--output-root", type=Path, default=Path("reports"))
    report_parser.add_argument("--smoothing-window", type=int, default=10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "evaluate":
        output = run_evaluation(
            args.checkpoint,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            output_root=args.output_root,
            device=args.device,
        )
    elif args.command == "random":
        output = run_random_baseline(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            output_root=args.output_root,
        )
    elif args.command == "compare":
        output = compare_evaluations(args.dqn, args.baseline, output_root=args.output_root)
    elif args.command == "plot":
        output = generate_plots(
            args.experiment,
            evaluation_directory=args.evaluation,
            baseline_directory=args.baseline,
            output_directory=args.output_directory,
            smoothing_window=args.smoothing_window,
        )
    else:
        output = generate_report(
            args.experiment,
            evaluation_directory=args.evaluation,
            baseline_directory=args.baseline,
            output_root=args.output_root,
            smoothing_window=args.smoothing_window,
        )
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
