from rl.environment.rewards import RewardConfig, compute_step_reward


def test_regular_pellet_reward() -> None:
    before = {"pelletsRemaining": 10, "lives": 3, "won": False, "gameOver": False, "score": 0.0}
    after = {"pelletsRemaining": 9, "lives": 3, "won": False, "gameOver": False, "score": 1.0, "lastPellet": {"type": "regular"}}
    assert compute_step_reward(before, after) == 1.0


def test_power_pellet_reward() -> None:
    before = {"pelletsRemaining": 10, "lives": 3, "won": False, "gameOver": False, "score": 0.0}
    after = {"pelletsRemaining": 9, "lives": 3, "won": False, "gameOver": False, "score": 3.0, "lastPellet": {"type": "power"}}
    assert compute_step_reward(before, after) == 3.0


def test_life_loss_and_win_rewards() -> None:
    before = {"pelletsRemaining": 5, "lives": 2, "won": False, "gameOver": False, "score": 10.0}
    after = {"pelletsRemaining": 0, "lives": 1, "won": True, "gameOver": False, "score": 60.0}
    reward = compute_step_reward(before, after)
    assert reward == 41.0


def test_no_duplicate_rewards() -> None:
    before = {"pelletsRemaining": 1, "lives": 3, "won": False, "gameOver": False, "score": 0.0}
    after = {"pelletsRemaining": 0, "lives": 3, "won": True, "gameOver": False, "score": 51.0}
    reward = compute_step_reward(before, after)
    assert reward == 50.0 + 1.0


def test_reward_config_is_configurable() -> None:
    config = RewardConfig(regular_pellet=2.0, power_pellet=7.0, maze_cleared=42.0)
    before = {"pelletsRemaining": 1, "lives": 3, "won": False, "gameOver": False, "score": 0.0}
    after = {"pelletsRemaining": 0, "lives": 3, "won": True, "gameOver": False, "score": 42.0}
    assert compute_step_reward(before, after, config) == 42.0 + 2.0


def test_ghost_capture_reward_uses_game_event_log() -> None:
    before = {"score": 3.0, "eventLog": []}
    after = {"score": 8.0, "eventLog": ["ghost-captured"]}
    assert compute_step_reward(before, after) == 5.0


def test_all_lives_lost_combines_life_and_terminal_penalties() -> None:
    before = {"lives": 1, "gameOver": False, "score": 0.0}
    after = {"lives": 0, "gameOver": True, "score": -30.0}
    assert compute_step_reward(before, after) == -30.0
