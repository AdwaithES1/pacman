from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    """Fixed-size NumPy replay storage with independently retained end flags."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if not observation_shape or any(size <= 0 for size in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions.")
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)
        self.states = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_states = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        if state_array.shape != self.observation_shape or next_state_array.shape != self.observation_shape:
            raise ValueError(f"States must both have shape {self.observation_shape}.")
        if not np.isfinite(state_array).all() or not np.isfinite(next_state_array).all():
            raise ValueError("States must contain only finite values.")
        if not isinstance(action, (int, np.integer)) or action < 0:
            raise ValueError("action must be a non-negative integer.")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite.")

        index = self.position
        self.states[index] = state_array
        self.next_states[index] = next_state_array
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError("batch_size must be positive and no greater than the current buffer size.")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return TransitionBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            terminated=self.terminated[indices].copy(),
            truncated=self.truncated[indices].copy(),
        )

    def state_dict(self) -> dict[str, Any]:
        indices = np.arange(self.size)
        return {
            "capacity": self.capacity,
            "observation_shape": self.observation_shape,
            "size": self.size,
            "position": self.position,
            "states": self.states[indices].copy(),
            "next_states": self.next_states[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "terminated": self.terminated[indices].copy(),
            "truncated": self.truncated[indices].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity or tuple(state["observation_shape"]) != self.observation_shape:
            raise ValueError("Replay-buffer checkpoint configuration does not match.")
        size = int(state["size"])
        if not 0 <= size <= self.capacity:
            raise ValueError("Replay-buffer checkpoint has an invalid size.")
        self.size = size
        self.position = int(state["position"])
        if not 0 <= self.position < self.capacity:
            raise ValueError("Replay-buffer checkpoint has an invalid position.")
        for name in ("states", "next_states", "actions", "rewards", "terminated", "truncated"):
            source = np.asarray(state[name])
            destination = getattr(self, name)
            expected_shape = (size, *self.observation_shape) if name in {"states", "next_states"} else (size,)
            if source.shape != expected_shape:
                raise ValueError(f"Replay-buffer checkpoint field {name!r} has an invalid shape.")
            destination[:size] = source


__all__ = ["ReplayBuffer", "TransitionBatch"]
