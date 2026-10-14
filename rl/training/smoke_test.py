from __future__ import annotations

import os
import random
import sys
from typing import Any, Mapping

import numpy as np
import requests

from rl.environment.pacman_env import PacmanEnv


SERVER_URL = os.getenv("PACMAN_BASE_URL", "http://127.0.0.1:3000")


def _get_server_status(url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{url}/api/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Health endpoint did not return a JSON object.")
        return payload
    except requests.RequestException as exc:
        raise RuntimeError(f"JavaScript Pac-Man server is unavailable at {url}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"JavaScript Pac-Man server responded with invalid health data: {exc}") from exc


def _pacman_state(state: Mapping[str, Any]) -> tuple[int, int] | None:
    entities = state.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, Mapping) and (entity.get("id") == "pacman" or entity.get("role") == "player"):
                x = entity.get("x")
                y = entity.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return int(x), int(y)
    pacman = state.get("pacman")
    if isinstance(pacman, Mapping):
        x = pacman.get("x")
        y = pacman.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return int(x), int(y)
    return None


def _ghost_positions(state: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    entities = state.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            if entity.get("role") != "ghost" and entity.get("id") not in {"blinky", "pinky", "inky", "clyde"}:
                continue
            x = entity.get("x")
            y = entity.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                result.append((int(x), int(y), str(entity.get("state") or "normal")))
    return result


def _validate_state_matches(env: PacmanEnv, state: Mapping[str, Any]) -> None:
    actual_state = env.game_client.get_state()
    if not isinstance(actual_state, Mapping):
        raise AssertionError("Actual game state from the server was not a mapping.")

    for key in ("score", "lives", "pelletsRemaining", "pelletsTotal", "won", "gameOver", "terminated", "truncated"):
        if state.get(key) != actual_state.get(key):
            raise AssertionError(
                f"Environment state mismatch for '{key}': env={state.get(key)} actual={actual_state.get(key)}"
            )

    env_pacman = _pacman_state(state)
    real_pacman = _pacman_state(actual_state)
    if env_pacman != real_pacman:
        raise AssertionError(f"Pac-Man position mismatch: env={env_pacman} actual={real_pacman}")

    env_ghosts = _ghost_positions(state)
    real_ghosts = _ghost_positions(actual_state)
    if env_ghosts != real_ghosts:
        raise AssertionError(f"Ghost state mismatch: env={env_ghosts} actual={real_ghosts}")


def run_smoke_test() -> None:
    server_url = SERVER_URL.rstrip("/")
    try:
        _get_server_status(server_url)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    env = PacmanEnv(base_url=server_url, timeout=5.0, max_steps=25)
    previous_reward = 0.0

    try:
        observation, info = env.reset()
        state = info.get("state", info)
        if not isinstance(state, Mapping):
            raise AssertionError("Reset response did not include a valid state mapping.")

        if observation.dtype != np.float32:
            raise AssertionError(f"Expected float32 observation, received {observation.dtype}.")
        if observation.shape != env.observation_space.shape:
            raise AssertionError(f"Observation shape mismatch: {observation.shape} vs {env.observation_space.shape}.")
        if not env.observation_space.contains(observation):
            raise AssertionError("Reset observation does not belong to the declared observation space.")

        pacman_position = _pacman_state(state)
        ghost_positions = _ghost_positions(state)
        print("Initial reset: shape=", observation.shape, "dtype=", observation.dtype)
        print("Observation in observation space:", env.observation_space.contains(observation))
        print("Pac-Man position:", pacman_position)
        print("Ghost positions:", ghost_positions)
        print("Score:", state.get("score"), "Pellets remaining:", state.get("pelletsRemaining"), "Lives:", state.get("lives"))

        for step_index in range(1, 101):
            action = random.randrange(env.action_space.n)
            state_before = env.game_client.get_state()
            observation, reward, terminated, truncated, info = env.step(action)
            state = info.get("state", info)

            if not isinstance(state, Mapping):
                raise AssertionError(f"Step #{step_index}: state payload was not a mapping.")
            if not env.observation_space.contains(observation):
                raise AssertionError(f"Step #{step_index}: observation is outside the declared space.")
            if not np.isfinite(reward):
                raise AssertionError(f"Step #{step_index}: reward is not finite: {reward!r}")
            if not isinstance(terminated, bool) or not isinstance(truncated, bool):
                raise AssertionError(f"Step #{step_index}: terminated/truncated must be boolean values.")

            _validate_state_matches(env, state)
            state_after = env.game_client.get_state()
            score_delta = int(state_after.get("score", 0) or 0) - int(state_before.get("score", 0) or 0)
            pellet_delta = int(state_before.get("pelletsRemaining", 0) or 0) - int(state_after.get("pelletsRemaining", 0) or 0)
            life_delta = int(state_before.get("lives", 0) or 0) - int(state_after.get("lives", 0) or 0)
            if previous_reward != 0.0 and reward == previous_reward and score_delta == 0 and pellet_delta == 0 and life_delta == 0:
                raise AssertionError(
                    f"Step #{step_index}: reward duplicated across consecutive steps without a corresponding state change: {previous_reward} -> {reward}"
                )

            if step_index % 10 == 0:
                print(
                    f"Step {step_index}: score={state.get('score')} pellets={state.get('pelletsRemaining')} "
                    f"lives={state.get('lives')} reward={reward} status={'terminated' if terminated else 'truncated' if truncated else 'running'}"
                )

            previous_reward = float(reward)

            if terminated or truncated:
                print(f"Episode ended at step {step_index}: terminated={terminated} truncated={truncated}")
                observation, info = env.reset()
                state = info.get("state", info)
                if not isinstance(state, Mapping):
                    raise AssertionError("Reset after terminal state returned invalid state data.")
                previous_reward = 0.0
                print("Reset complete. Pac-Man position:", _pacman_state(state), "Lives:", state.get("lives"))

        print("Smoke test completed successfully after 100 actions.")
    finally:
        env.close()
        print("Environment closed successfully.")


if __name__ == "__main__":
    run_smoke_test()
