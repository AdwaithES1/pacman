import csv
import json

from rl.training.metrics import EpisodeMetricsLogger


def test_episode_metrics_logger_writes_config_and_distinct_score_return(tmp_path):
    logger = EpisodeMetricsLogger(tmp_path, {"seed": 7, "bootstrap_on_truncation": True})
    logger.log_episode(
        {
            "episode": 1,
            "global_env_step": 25,
            "episode_return": -3.0,
            "game_score": -13.0,
            "episode_length": 25,
            "pellets_remaining": 240,
            "terminated": False,
            "truncated": True,
            "epsilon": 0.8,
            "latest_loss": 0.25,
            "gradient_updates": 10,
        }
    )
    logger.close()

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 7
    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as metrics_file:
        row = next(csv.DictReader(metrics_file))
    assert row["episode_return"] == "-3.0"
    assert row["game_score"] == "-13.0"
    assert row["terminated"] == "False"
    assert row["truncated"] == "True"


def test_episode_metrics_logger_does_not_overwrite_existing_files(tmp_path):
    EpisodeMetricsLogger(tmp_path, {"seed": 1}).close()
    try:
        EpisodeMetricsLogger(tmp_path, {"seed": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("Logger unexpectedly overwrote an existing experiment.")
