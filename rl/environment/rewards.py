from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardConfig:
    regular_pellet: float = 1.0
    power_pellet: float = 3.0
    ghost_capture: float = 5.0
    life_lost: float = -10.0
    maze_cleared: float = 50.0
    all_lives_lost: float = -20.0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def compute_step_reward(before_state: Mapping[str, Any], after_state: Mapping[str, Any], config: RewardConfig | None = None) -> float:
    """Compute total reward for a game transition using event-based rules.

    Rewards are cumulative within a single step if more than one event occurs.
    """
    reward_cfg = config or RewardConfig()
    before = _as_mapping(before_state)
    after = _as_mapping(after_state)

    reward = 0.0

    before_pellets = int(before.get("pelletsRemaining", 0))
    after_pellets = int(after.get("pelletsRemaining", 0))
    if after_pellets < before_pellets:
        last_pellet = _as_mapping(after.get("lastPellet"))
        pellet_type = str(last_pellet.get("type", "regular"))
        reward += reward_cfg.power_pellet if pellet_type == "power" else reward_cfg.regular_pellet

    before_lives = int(before.get("lives", 3))
    after_lives = int(after.get("lives", 3))
    if after_lives < before_lives:
        reward += reward_cfg.life_lost

    if bool(after.get("won")) and not bool(before.get("won", False)):
        reward += reward_cfg.maze_cleared

    if bool(after.get("gameOver")) and not bool(before.get("gameOver", False)):
        reward += reward_cfg.all_lives_lost

    before_events = before.get("eventLog", [])
    after_events = after.get("eventLog", [])
    captured_ghost = (
        isinstance(before_events, list)
        and isinstance(after_events, list)
        and len(after_events) > len(before_events)
        and "ghost-captured" in after_events[len(before_events):]
    )
    before_score = float(before.get("score", 0.0))
    after_score = float(after.get("score", 0.0))
    if captured_ghost or (after_score > before_score and (after_score - before_score) >= 200):
        reward += reward_cfg.ghost_capture

    return float(reward)
