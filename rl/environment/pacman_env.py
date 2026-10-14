from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from .game_client import GameClient
from .observations import build_observation
from .rewards import compute_step_reward


class PacmanEnv(gym.Env):
    """Gymnasium-compatible Pac-Man environment backed by the local JS game."""

    metadata = {"render_modes": []}

    ACTION_MAP = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    def __init__(self, base_url: str | None = None, timeout: float = 5.0, max_steps: int = 200) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_steps = max_steps
        self.game_client = GameClient(base_url=base_url, timeout=timeout)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, 31, 28),
            dtype=np.float32,
        )
        self.terminated = False
        self.truncated = False
        self.episode_return = 0.0
        self.episode_length = 0
        self._last_state: Mapping[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._maze_rows: list[str] | None = None

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        if self._maze_rows is None:
            maze = self.game_client.get_maze()
            rows = maze.get("rows")
            if not isinstance(rows, list) or not rows or not all(isinstance(row, str) for row in rows):
                raise ValueError("Maze response did not include valid canonical rows.")
            self._maze_rows = rows
        payload = self.game_client.reset(max_steps=self.max_steps)
        state = payload.get("state", payload)
        if not isinstance(state, Mapping):
            raise ValueError("Reset response did not include a valid state payload.")
        state = dict(state)
        maze_state = state.get("maze")
        state["maze"] = {
            **(maze_state if isinstance(maze_state, Mapping) else {}),
            "rows": self._maze_rows,
        }
        self.terminated = False
        self.truncated = False
        self.episode_return = 0.0
        self.episode_length = 0
        self._last_state = state
        game_score = float(state.get("score", 0.0))
        self._last_info = {
            "state": state,
            "reset": True,
            "score": game_score,
            "game_score": game_score,
            "reward": 0.0,
            "step_reward": 0.0,
            "episode_return": self.episode_return,
            "cumulative_reward": self.episode_return,
        }
        observation = build_observation(state)
        if not self.observation_space.contains(observation):
            raise ValueError(
                f"Built observation does not match the declared space: {observation.shape}."
            )
        return observation, self._last_info.copy()

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is finished. Call reset() before stepping again.")
        action_value = int(action)
        if action_value not in self.ACTION_MAP:
            raise ValueError("Action must be one of 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.")

        before_state = self._last_state if isinstance(self._last_state, Mapping) else {}
        response = self.game_client.step(action_value)
        state = response.get("state", response)
        if not isinstance(state, Mapping):
            raise ValueError("Step response did not include a valid state payload.")

        observation = build_observation(state)
        if not self.observation_space.contains(observation):
            raise ValueError(
                f"Built observation does not match the declared space: {observation.shape}."
            )
        step_reward = float(response.get("reward", 0.0))
        if before_state:
            computed_reward = float(compute_step_reward(before_state, state))
            if np.isfinite(computed_reward) and (not np.isfinite(step_reward) or np.isclose(step_reward, 0.0, atol=1e-9) != np.isclose(computed_reward, 0.0, atol=1e-9)):
                step_reward = computed_reward
        terminated = bool(response.get("terminated", False) or state.get("terminated", False) or state.get("won", False) or state.get("gameOver", False))
        truncated = bool(response.get("truncated", False) or state.get("truncated", False))
        self.terminated = terminated
        self.truncated = truncated
        self.episode_length += 1
        self.episode_return += step_reward
        self._last_state = state

        game_score = float(state.get("score", 0.0))
        info = {
            "state": state,
            "action": action_value,
            "action_name": self.ACTION_MAP[action_value],
            "step": response.get("info", {}),
            "score": game_score,
            "game_score": game_score,
            "reward": step_reward,
            "step_reward": step_reward,
            "episode_return": self.episode_return,
            "cumulative_reward": self.episode_return,
            "terminated": terminated,
            "truncated": truncated,
        }
        if truncated and not terminated:
            info["truncation_reason"] = "max_steps"
        self._last_info = info
        return observation, step_reward, terminated, truncated, info

    def close(self):
        self.game_client.close()


__all__ = ["PacmanEnv"]
