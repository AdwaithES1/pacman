from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_update_interval: int = 500
    gradient_clip: float = 10.0
    hidden_size: int = 256
    conv_channels: tuple[int, ...] = (32, 64, 64)
    seed: int = 7
    device: str = "auto"


class QNetwork(nn.Module):
    """CNN mapping normalized CHW maze observations to action values."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_count: int,
        conv_channels: tuple[int, ...] = (32, 64, 64),
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3 or min(observation_shape) <= 0:
            raise ValueError("observation_shape must be a positive (channels, height, width) tuple.")
        if action_count <= 0:
            raise ValueError("action_count must be positive.")
        if not conv_channels or any(channel <= 0 for channel in conv_channels):
            raise ValueError("conv_channels must contain positive channel sizes.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        self.observation_shape = tuple(int(size) for size in observation_shape)
        self.action_count = int(action_count)
        layers: list[nn.Module] = []
        in_channels = self.observation_shape[0]
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                ]
            )
            in_channels = out_channels
        layers.extend([nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_channels * 4 * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim not in (3, 4):
            raise ValueError("Expected a CHW observation or a BCHW batch of observations.")
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if tuple(observations.shape[1:]) != self.observation_shape:
            raise ValueError(
                f"Expected observations shaped (batch, {self.observation_shape}), "
                f"received {tuple(observations.shape)}."
            )
        observations = observations.to(dtype=torch.float32)
        if not torch.isfinite(observations).all():
            raise ValueError("Observations must contain only finite values.")
        if torch.any(observations < 0.0) or torch.any(observations > 1.0):
            raise ValueError("Observations must be normalized to the [0, 1] range.")
        q_values = self.head(self.features(observations))
        if q_values.shape[-1] != self.action_count:
            raise RuntimeError("Q-network output dimension does not match the action count.")
        return q_values


class DQNAgent:
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_count: int,
        config: DQNConfig | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        if self.config.batch_size <= 0 or self.config.replay_capacity < self.config.batch_size:
            raise ValueError("replay_capacity must be at least batch_size, and batch_size must be positive.")
        if self.config.warmup_steps < 0 or self.config.target_update_interval <= 0:
            raise ValueError("warmup_steps must be non-negative and target_update_interval must be positive.")
        if not 0.0 <= self.config.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if self.config.learning_rate <= 0 or self.config.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive.")
        if not 0.0 <= self.config.epsilon_end <= self.config.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1.")
        if self.config.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive.")

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self._random = random.Random(self.config.seed)
        self.device = self._resolve_device(self.config.device)
        self.observation_shape = tuple(observation_shape)
        self.action_count = int(action_count)
        self.online_network = QNetwork(
            self.observation_shape,
            self.action_count,
            self.config.conv_channels,
            self.config.hidden_size,
        ).to(self.device)
        self.target_network = QNetwork(
            self.observation_shape,
            self.action_count,
            self.config.conv_channels,
            self.config.hidden_size,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(
            self.config.replay_capacity,
            self.observation_shape,
            seed=self.config.seed,
        )
        self.global_step = 0
        self.completed_episodes = 0
        self.gradient_updates = 0
        self.latest_loss: float | None = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return resolved

    def epsilon(self, step: int | None = None) -> float:
        progress = min(max(self.global_step if step is None else step, 0) / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(
        self,
        observation: np.ndarray,
        *,
        explore: bool = True,
        epsilon: float | None = None,
    ) -> int:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != self.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.observation_shape}, received {observation_array.shape}."
            )
        if not np.isfinite(observation_array).all() or np.any(observation_array < 0) or np.any(observation_array > 1):
            raise ValueError("Observations must be finite and normalized to the [0, 1] range.")
        exploration_rate = self.epsilon() if epsilon is None else epsilon
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        if explore and self._random.random() < exploration_rate:
            return self._random.randrange(self.action_count)

        observation_tensor = torch.from_numpy(observation_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online_network(observation_tensor).argmax(dim=1).item())

    @staticmethod
    def compute_td_targets(
        rewards: torch.Tensor,
        next_q_values: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Bootstrap at truncations; mask only true terminal transitions."""
        return rewards + gamma * (1.0 - terminated.float()) * next_q_values

    def learn(self) -> float | None:
        if self.global_step < self.config.warmup_steps:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.float32, device=self.device)

        predicted_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            targets = self.compute_td_targets(rewards, next_q, terminated, self.config.gamma)
        loss = F.smooth_l1_loss(predicted_q, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.gradient_updates += 1
        self.latest_loss = float(loss.item())
        if self.gradient_updates % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return self.latest_loss

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "observation_shape": self.observation_shape,
            "action_count": self.action_count,
            "config": asdict(self.config),
            "global_step": self.global_step,
            "completed_episodes": self.completed_episodes,
            "gradient_updates": self.gradient_updates,
            "latest_loss": self.latest_loss,
            "replay_buffer": self.replay_buffer.state_dict(),
            "python_rng_state": self._random.getstate(),
            "numpy_rng_state": self.replay_buffer.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        observation_shape: tuple[int, int, int],
        action_count: int,
        device: str | None = None,
    ) -> DQNAgent:
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        saved_shape = tuple(checkpoint["observation_shape"])
        if saved_shape != tuple(observation_shape) or int(checkpoint["action_count"]) != action_count:
            raise ValueError("Checkpoint observation or action space does not match the current environment.")
        config_values = dict(checkpoint["config"])
        config_values["conv_channels"] = tuple(config_values["conv_channels"])
        if device is not None:
            config_values["device"] = device
        agent = cls(observation_shape, action_count, DQNConfig(**config_values))
        agent.online_network.load_state_dict(checkpoint["online_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.global_step = int(checkpoint["global_step"])
        agent.completed_episodes = int(checkpoint["completed_episodes"])
        agent.gradient_updates = int(checkpoint["gradient_updates"])
        agent.latest_loss = checkpoint["latest_loss"]
        agent.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        agent._random.setstate(checkpoint["python_rng_state"])
        agent.replay_buffer.rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return agent


__all__ = ["DQNAgent", "DQNConfig", "QNetwork"]
