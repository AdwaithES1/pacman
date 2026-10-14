"""Environment and client helpers for the Pac-Man RL integration."""

from .game_client import GameClient, PacmanGameClient
from .observations import build_observation, OBSERVATION_CHANNELS, OBSERVATION_SHAPE
from .pacman_env import PacmanEnv
from .rewards import RewardConfig, compute_step_reward

__all__ = [
    "GameClient",
    "PacmanEnv",
    "PacmanGameClient",
    "RewardConfig",
    "build_observation",
    "compute_step_reward",
    "OBSERVATION_CHANNELS",
    "OBSERVATION_SHAPE",
]
