from __future__ import annotations

import csv
import json
import platform
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import gymnasium
import numpy
import torch


EPISODE_FIELDS = (
    "episode",
    "global_env_step",
    "episode_return",
    "game_score",
    "pellets_collected",
    "pellets_remaining",
    "episode_length",
    "duration_seconds",
    "lives_lost",
    "won",
    "terminated",
    "truncated",
    "termination_reason",
    "epsilon",
    "latest_loss",
    "gradient_updates",
)

TRAINING_FIELDS = (
    "global_env_step",
    "episodes_completed",
    "window_episodes",
    "mean_episode_return",
    "mean_game_score",
    "mean_pellets_collected",
    "mean_pellets_remaining",
    "completion_rate",
    "mean_episode_length",
    "latest_loss",
    "epsilon",
    "learning_rate",
    "gradient_updates",
    "steps_per_second",
    "elapsed_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def software_versions() -> dict[str, str | None]:
    try:
        torch_version = torch.__version__
    except AttributeError:
        torch_version = None
    try:
        gymnasium_version = gymnasium.__version__
    except AttributeError:
        gymnasium_version = None
    try:
        numpy_version = numpy.__version__
    except AttributeError:
        numpy_version = None
    try:
        matplotlib_version = version("matplotlib")
    except PackageNotFoundError:
        matplotlib_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "gymnasium": gymnasium_version,
        "numpy": numpy_version,
        "matplotlib": matplotlib_version,
    }


def create_run_directory(output_root: str | Path, prefix: str, seed: int) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment_id = f"{prefix}-{timestamp}-seed{seed}-{uuid.uuid4().hex[:8]}"
    run_directory = root / experiment_id
    run_directory.mkdir()
    return run_directory


class EpisodeMetricsLogger:
    def __init__(
        self,
        run_directory: str | Path,
        configuration: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_directory / "episodes.csv"
        self.config_path = self.run_directory / "config.json"
        self.training_path = self.run_directory / "training.csv"
        self.metadata_path = self.run_directory / "run_metadata.json"
        with self.config_path.open("x", encoding="utf-8") as config_file:
            json.dump(dict(configuration), config_file, indent=2, sort_keys=True)
            config_file.write("\n")
        run_metadata = dict(metadata or {})
        run_metadata.setdefault("experiment_id", self.run_directory.name)
        run_metadata.setdefault("started_at", utc_now())
        run_metadata.setdefault("status", "running")
        run_metadata.setdefault("software_versions", software_versions())
        run_metadata.setdefault("device", configuration.get("resolved_device", "unknown"))
        with self.metadata_path.open("x", encoding="utf-8") as metadata_file:
            json.dump(run_metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        self._episode_file = self.csv_path.open("x", newline="", encoding="utf-8")
        self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=EPISODE_FIELDS)
        self._episode_writer.writeheader()
        self._episode_file.flush()
        self._training_file = self.training_path.open("x", newline="", encoding="utf-8")
        self._training_writer = csv.DictWriter(self._training_file, fieldnames=TRAINING_FIELDS)
        self._training_writer.writeheader()
        self._training_file.flush()
        self._closed = False

    def log_episode(self, metrics: Mapping[str, Any]) -> None:
        self._episode_writer.writerow({field: metrics.get(field) for field in EPISODE_FIELDS})
        self._episode_file.flush()

    def log_training(self, metrics: Mapping[str, Any]) -> None:
        self._training_writer.writerow({field: metrics.get(field) for field in TRAINING_FIELDS})
        self._training_file.flush()

    def finish(self, status: str, **details: Any) -> None:
        if status not in {"completed", "interrupted", "failed"}:
            raise ValueError("status must be completed, interrupted, or failed.")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata.update(details)
        metadata["status"] = status
        metadata["finished_at"] = utc_now()
        with self.metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")

    def close(self) -> None:
        if not self._closed:
            self._episode_file.close()
            self._training_file.close()
            self._closed = True


__all__ = [
    "EPISODE_FIELDS",
    "TRAINING_FIELDS",
    "EpisodeMetricsLogger",
    "create_run_directory",
    "software_versions",
    "utc_now",
]
