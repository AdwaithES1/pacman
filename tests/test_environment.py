import numpy as np
import pytest

from rl.environment import PacmanEnv


def test_observation_shape_and_space():
    env = PacmanEnv(base_url="http://127.0.0.1:3000", timeout=1.0)
    try:
        obs, info = env.reset()
        assert obs.shape == (6, 31, 28)
        assert env.observation_space.contains(obs)
        assert obs[0].sum() > 0
        assert obs[1].sum() + obs[2].sum() > 0
        assert obs[3].sum() == 1
        assert obs[4].sum() + obs[5].sum() == 4
        assert isinstance(info, dict)
    except RuntimeError:
        pytest.skip("Pac-Man server is not running; environment tests require the local JS server.")
    finally:
        env.close()


def test_action_mapping_is_valid():
    env = PacmanEnv(base_url="http://127.0.0.1:3000", timeout=1.0)
    try:
        env.reset()
        for action in range(4):
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            assert np.isfinite(reward)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated or truncated:
                break
    except RuntimeError:
        pytest.skip("Pac-Man server is not running; environment tests require the local JS server.")
    finally:
        env.close()


def test_invalid_action_is_rejected():
    env = PacmanEnv(base_url="http://127.0.0.1:3000", timeout=1.0)
    try:
        env.reset()
        with pytest.raises((ValueError, RuntimeError)):
            env.step(99)
    except RuntimeError:
        pytest.skip("Pac-Man server is not running; environment tests require the local JS server.")
    finally:
        env.close()


def test_step_metrics_separate_game_score_and_episode_return():
    env = PacmanEnv(base_url="http://127.0.0.1:3000", timeout=1.0)
    try:
        _, info = env.reset()
        assert info["score"] == info["game_score"]
        assert info["episode_return"] == 0.0
        _, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        assert info["reward"] == reward
        assert info["step_reward"] == reward
        assert info["score"] == info["game_score"]
        assert info["episode_return"] >= reward
        assert info["episode_return"] == info["cumulative_reward"]
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    except RuntimeError:
        pytest.skip("Pac-Man server is not running; environment tests require the local JS server.")
    finally:
        env.close()
