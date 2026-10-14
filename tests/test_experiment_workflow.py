import csv
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl.agents import DQNAgent, DQNConfig
from rl.training import experiment_workflow as workflow
from rl.training import train as training

torch.set_num_threads(1)


OBSERVATION_SHAPE = (6, 31, 28)


class TinyGameEnv:
    def __init__(self, max_steps=2):
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(0, 1, shape=OBSERVATION_SHAPE, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.game_client = SimpleNamespace(base_url="http://127.0.0.1:3000")
        self.step_count = 0
        self.episode = 0
        self.closed = False

    @staticmethod
    def _state(score=0, pellets=10, lives=3, won=False):
        return {
            "score": score,
            "pelletsRemaining": pellets,
            "pelletsTotal": 10,
            "lives": lives,
            "won": won,
            "gameOver": False,
        }

    def reset(self, seed=None):
        self.episode += 1
        self.step_count = 0
        state = self._state()
        return np.zeros(OBSERVATION_SHAPE, dtype=np.float32), {"state": state}

    def step(self, action):
        assert self.action_space.contains(action)
        self.step_count += 1
        state = self._state(score=8, pellets=9, lives=3)
        truncated = self.step_count >= self.max_steps
        info = {"state": state, "game_score": state["score"]}
        return np.ones(OBSERVATION_SHAPE, dtype=np.float32), 1.5, False, truncated, info

    def close(self):
        self.closed = True


def _small_agent_config():
    return DQNConfig(
        batch_size=2,
        replay_capacity=8,
        warmup_steps=0,
        hidden_size=8,
        conv_channels=(4,),
        device="cpu",
    )


def _episode_dir(tmp_path: Path, name: str, *, limit=2, return_value=3.0, score=6.0):
    directory = tmp_path / name
    directory.mkdir()
    config = {
        "mode": "evaluation",
        "seed": 17,
        "max_steps_per_episode": limit,
        "reward_configuration": {
            "regular_pellet": 1.0,
            "power_pellet": 3.0,
            "ghost_capture": 5.0,
            "life_lost": -10.0,
            "maze_cleared": 50.0,
            "all_lives_lost": -20.0,
        },
        "observation_shape": list(OBSERVATION_SHAPE),
        "action_count": 4,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    fields = (
        "episode", "global_env_step", "episode_return", "game_score",
        "pellets_collected", "pellets_remaining", "episode_length",
        "duration_seconds", "lives_lost", "won", "terminated", "truncated",
        "termination_reason", "epsilon", "latest_loss", "gradient_updates",
    )
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for episode in range(1, 3):
            writer.writerow({
                "episode": episode,
                "global_env_step": episode * limit,
                "episode_return": return_value,
                "game_score": score,
                "pellets_collected": 1,
                "pellets_remaining": 9,
                "episode_length": limit,
                "duration_seconds": 0.01,
                "lives_lost": 0,
                "won": False,
                "terminated": False,
                "truncated": True,
                "termination_reason": "max_steps",
                "epsilon": 0.0,
                "latest_loss": "",
                "gradient_updates": 0,
            })
    (directory / "summary.json").write_text(
        json.dumps({"status": "completed", **workflow.summarize_episodes(workflow._read_csv(directory / "episodes.csv"))}),
        encoding="utf-8",
    )
    return directory


def test_evaluation_does_not_update_network_or_optimizer(tmp_path, monkeypatch):
    agent = DQNAgent(OBSERVATION_SHAPE, 4, _small_agent_config())
    checkpoint = tmp_path / "current-format.pt"
    agent.save_checkpoint(checkpoint)
    loaded = DQNAgent.load_checkpoint(checkpoint, OBSERVATION_SHAPE, 4, device="cpu")
    online_before = [parameter.detach().clone() for parameter in loaded.online_network.parameters()]
    optimizer_before = json.dumps(loaded.optimizer.state_dict(), default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value), sort_keys=True)
    fake_env = TinyGameEnv()
    monkeypatch.setattr(workflow, "PacmanEnv", lambda max_steps: fake_env)
    monkeypatch.setattr(DQNAgent, "load_checkpoint", classmethod(lambda cls, *args, **kwargs: loaded))

    output = workflow.run_evaluation(checkpoint, episodes=2, max_steps=2, output_root=tmp_path / "evals")

    assert fake_env.closed
    assert len(workflow._read_csv(output / "episodes.csv")) == 2
    assert workflow._read_json(output / "summary.json")["episodes"] == 2
    assert all(torch.equal(before, after) for before, after in zip(online_before, loaded.online_network.parameters()))
    optimizer_after = json.dumps(loaded.optimizer.state_dict(), default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value), sort_keys=True)
    assert optimizer_before == optimizer_after
    rows = workflow._read_csv(output / "episodes.csv")
    assert rows[0]["episode_return"] == "3.0"
    assert rows[0]["game_score"] == "8"
    assert rows[0]["epsilon"] == "0.0"


def test_random_baseline_logs_real_environment_metrics(tmp_path, monkeypatch):
    fake_env = TinyGameEnv()
    monkeypatch.setattr(workflow, "PacmanEnv", lambda max_steps: fake_env)
    output = workflow.run_random_baseline(episodes=2, max_steps=2, output_root=tmp_path, seed=8)
    rows = workflow._read_csv(output / "episodes.csv")
    summary = workflow._read_json(output / "summary.json")
    assert fake_env.closed
    assert len(rows) == 2
    assert all(row["episode_return"] == "3.0" for row in rows)
    assert all(row["game_score"] == "8" for row in rows)
    assert summary["completion_rate"] is None
    assert summary["completion_observations"] == 0
    assert summary["truncated_episodes"] == 2


def test_training_run_records_periodic_metrics_and_completion_metadata(tmp_path, monkeypatch):
    fake_env = TinyGameEnv()
    monkeypatch.setattr(training, "PacmanEnv", lambda max_steps: fake_env)
    output = training.train(
        total_steps=4,
        max_steps=2,
        output_root=tmp_path / "runs",
        config=_small_agent_config(),
    )
    episode_rows = workflow._read_csv(output / "episodes.csv")
    periodic_rows = workflow._read_csv(output / "training.csv")
    metadata = workflow._read_json(output / "run_metadata.json")
    configuration = workflow._read_json(output / "config.json")
    assert fake_env.closed
    assert len(episode_rows) == 2
    assert episode_rows[0]["episode_return"] == "3.0"
    assert episode_rows[0]["game_score"] == "8"
    assert episode_rows[0]["pellets_collected"] == "1"
    assert len(periodic_rows) == 2
    assert periodic_rows[-1]["gradient_updates"] == "3"
    assert float(periodic_rows[-1]["latest_loss"]) >= 0
    assert configuration["dqn"]["batch_size"] == 2
    assert configuration["dqn"]["warmup_steps"] == 0
    assert configuration["dqn"]["device"] == "cpu"
    assert periodic_rows[-1]["completion_rate"] == ""
    assert metadata["status"] == "completed"
    assert metadata["global_env_step"] == 4
    assert (output / "checkpoint.pt").is_file()
    assert metadata["elapsed_seconds"] > 0


def test_training_saves_and_evaluates_stage_checkpoints(tmp_path, monkeypatch):
    fake_env = TinyGameEnv(max_steps=2)
    monkeypatch.setattr(training, "PacmanEnv", lambda max_steps: fake_env)
    monkeypatch.setattr(workflow, "PacmanEnv", lambda max_steps: fake_env)
    output = training.train(
        total_steps=4,
        max_steps=2,
        output_root=tmp_path / "runs",
        config=_small_agent_config(),
        checkpoint_every_steps=2,
        evaluation_every_steps=2,
        checkpoint_evaluation_episodes=1,
        evaluation_seed=31,
    )

    metadata = workflow._read_json(output / "run_metadata.json")
    assert [item["global_env_step"] for item in metadata["checkpoints"]] == [2, 4]
    assert [item["global_env_step"] for item in metadata["checkpoint_evaluations"]] == [2, 4]
    for stage in metadata["checkpoint_evaluations"]:
        evaluation_path = Path(stage["evaluation_directory"])
        evaluation_config = workflow._read_json(evaluation_path / "config.json")
        summary = workflow._read_json(evaluation_path / "summary.json")
        assert evaluation_config["evaluation_episodes"] == 1
        assert evaluation_config["seed"] == 31
        assert evaluation_config["training_checkpoint"]["global_env_step"] == stage["global_env_step"]
        assert evaluation_config["training_checkpoint"]["gradient_updates"] == stage["gradient_updates"]
        assert summary["episodes"] == 1
        assert (Path(stage["checkpoint"])).is_file()
    assert (output / "checkpoint.pt").is_file()


def test_training_evaluates_final_checkpoint_when_target_is_mid_episode(tmp_path, monkeypatch):
    fake_env = TinyGameEnv(max_steps=2)
    monkeypatch.setattr(training, "PacmanEnv", lambda max_steps: fake_env)
    monkeypatch.setattr(workflow, "PacmanEnv", lambda max_steps: fake_env)
    output = training.train(
        total_steps=5,
        max_steps=2,
        output_root=tmp_path / "runs",
        config=_small_agent_config(),
        checkpoint_every_steps=2,
        evaluation_every_steps=5,
        checkpoint_evaluation_episodes=1,
        evaluation_seed=31,
    )

    metadata = workflow._read_json(output / "run_metadata.json")
    training_rows = workflow._read_csv(output / "training.csv")
    assert metadata["global_env_step"] == 5
    assert metadata["partial_episode_steps"] == 1
    assert metadata["epsilon"] == pytest.approx(1.0 - 5 / 20_000 * 0.95)
    assert len(metadata["checkpoint_evaluations"]) == 1
    stage = metadata["checkpoint_evaluations"][0]
    assert stage["global_env_step"] == 5
    assert Path(stage["checkpoint"]).is_file()
    assert training_rows[-1]["global_env_step"] == "5"
    assert training_rows[-1]["window_episodes"] == "0"


def test_experiment_comparison_reports_conditions_and_separate_scores(tmp_path):
    dqn = _episode_dir(tmp_path, "dqn", return_value=2.0, score=9.0)
    random = _episode_dir(tmp_path, "random", return_value=1.0, score=4.0)
    output = workflow.compare_evaluations(dqn, random, output_root=tmp_path / "compare")
    result = workflow._read_json(output / "comparison.json")
    assert result["conditions_match"]
    assert result["dqn"]["episode_return"]["mean"] == 2.0
    assert result["dqn"]["game_score"]["mean"] == 9.0
    assert result["random_baseline"]["episode_return"]["mean"] == 1.0
    assert "No comparative performance claim" in result["performance_claim"]


def test_experiment_comparison_surfaces_episode_limit_differences(tmp_path):
    dqn = _episode_dir(tmp_path, "dqn", limit=2)
    random = _episode_dir(tmp_path, "random", limit=3)
    output = workflow.compare_evaluations(dqn, random, output_root=tmp_path / "compare")
    result = workflow._read_json(output / "comparison.json")
    assert not result["conditions_match"]
    assert result["condition_differences"]["max_steps_per_episode"] == {"dqn": 2, "random": 3}


def test_plots_generate_png_and_pdf_and_handle_empty_series(tmp_path):
    experiment = _episode_dir(tmp_path, "training")
    (experiment / "training.csv").write_text(
        "global_env_step,episodes_completed,window_episodes,mean_episode_return,mean_game_score,mean_pellets_collected,mean_pellets_remaining,completion_rate,mean_episode_length,latest_loss,epsilon,learning_rate,gradient_updates,steps_per_second,elapsed_seconds\n"
        "2,1,1,3,6,1,9,0,2,0.5,0.9,0.0001,1,3,0.6\n",
        encoding="utf-8",
    )
    output = workflow.generate_plots(experiment, output_directory=tmp_path / "plots", smoothing_window=2)
    manifest = workflow._read_json(output / "plots.json")
    assert "episode_return.png" in manifest["files"]
    assert "dqn_vs_random.pdf" in manifest["files"]
    assert all((output / name).is_file() for name in manifest["files"])


def test_report_generates_preliminary_markdown_with_relative_plot_links(tmp_path):
    experiment = _episode_dir(tmp_path, "training")
    (experiment / "config.json").write_text(
        json.dumps({
            "mode": "train",
            "seed": 17,
            "max_steps_per_episode": 2,
            "observation_shape": list(OBSERVATION_SHAPE),
            "action_count": 4,
            "network_architecture": {"type": "CNN", "convolution_channels": [4], "hidden_size": 8},
            "dqn": {"gamma": 0.99},
            "reward_configuration": {"regular_pellet": 1.0},
        }),
        encoding="utf-8",
    )
    (experiment / "run_metadata.json").write_text(
        json.dumps({"experiment_id": "synthetic-fixture", "status": "completed", "gradient_updates": 1}),
        encoding="utf-8",
    )
    output = workflow.generate_report(experiment, output_root=tmp_path / "reports")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Preliminary" in report
    assert "episode_return" in report
    assert "Actual game score" in report
    assert "[episode_return.png](plots/episode_return.png)" in report
    assert (output / "plots" / "episode_return.png").is_file()


def test_summary_preserves_missing_values_without_imputation():
    summary = workflow.summarize_episodes([
        {"episode_return": 2.0, "game_score": "", "won": "", "episode_length": 4},
        {"episode_return": 1.0, "episode_length": 5},
    ])
    assert summary["episode_return"]["mean"] == 1.5
    assert summary["game_score"]["mean"] is None
    assert summary["completion_rate"] is None
    assert summary["environment_steps"] == 9


def test_completion_rate_ignores_time_limit_truncations():
    summary = workflow.summarize_episodes([
        {"won": "True", "terminated": "True", "truncated": "False"},
        {"won": "False", "terminated": "True", "truncated": "False"},
        {"won": "False", "terminated": "False", "truncated": "True"},
    ])
    assert summary["completion_rate"] == 0.5
    assert summary["completion_observations"] == 2
    assert summary["wins_observed"] == 1
