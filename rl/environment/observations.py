from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

DEFAULT_WIDTH = 28
DEFAULT_HEIGHT = 31
OBSERVATION_CHANNELS = (
    "walls",
    "regular_pellets",
    "power_pellets",
    "pacman",
    "dangerous_ghosts",
    "frightened_ghosts",
)
OBSERVATION_SHAPE = (len(OBSERVATION_CHANNELS), DEFAULT_HEIGHT, DEFAULT_WIDTH)


def _grid_from_state(state: Mapping[str, Any]) -> tuple[int, int, np.ndarray | None]:
    maze = state.get("maze") or {}
    if isinstance(maze, Mapping):
        rows = maze.get("rows") or maze.get("pelletRows") or state.get("pelletRows")
        if isinstance(rows, list) and rows and all(isinstance(row, str) for row in rows):
            height = len(rows)
            width = max(len(row) for row in rows)
            grid = np.zeros((height, width), dtype=np.float32)
            for y, row in enumerate(rows):
                for x, value in enumerate(row):
                    if value == "#":
                        grid[y, x] = 1.0
            return height, width, grid

    pellet_rows = state.get("pelletRows")
    if isinstance(pellet_rows, list) and pellet_rows and all(isinstance(row, str) for row in pellet_rows):
        height = len(pellet_rows)
        width = max(len(row) for row in pellet_rows)
        grid = np.zeros((height, width), dtype=np.float32)
        for y, row in enumerate(pellet_rows):
            for x, value in enumerate(row):
                if value == "#":
                    grid[y, x] = 1.0
        return height, width, grid

    return DEFAULT_HEIGHT, DEFAULT_WIDTH, None


def _position_from_entity(entity: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(entity, Mapping):
        return None
    tile = entity.get("tile")
    if isinstance(tile, Mapping):
        x = tile.get("x")
        y = tile.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return int(x), int(y)
    x = entity.get("x")
    y = entity.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return int(x), int(y)
    return None


def _mark_positions(array: np.ndarray, positions: Iterable[tuple[int, int]], value: float) -> None:
    for x, y in positions:
        if 0 <= y < array.shape[1] and 0 <= x < array.shape[2]:
            array[0 if value == 1.0 else 0, y, x] = value


def build_observation(state: Mapping[str, Any]) -> np.ndarray:
    """Construct a deterministic multi-channel observation for the Pac-Man grid."""
    if not isinstance(state, Mapping):
        raise ValueError("Observation state must be a mapping-like object.")

    height, width, walls = _grid_from_state(state)
    observation = np.zeros((len(OBSERVATION_CHANNELS), height, width), dtype=np.float32)

    if walls is not None:
        observation[0] = walls

    pellet_rows = state.get("pelletRows") or state.get("maze", {}).get("rows") or []
    if isinstance(pellet_rows, list) and pellet_rows and all(isinstance(row, str) for row in pellet_rows):
        for y, row in enumerate(pellet_rows[:height]):
            for x, tile in enumerate(row[:width]):
                if tile == ".":
                    observation[1, y, x] = 1.0
                elif tile == "o":
                    observation[2, y, x] = 1.0

    pacman = None
    entities = state.get("entities")
    if isinstance(entities, list):
        pacman = next((entity for entity in entities if entity.get("id") == "pacman"), None)
    if pacman is None:
        pacman = state.get("pacman")
    pacman_position = _position_from_entity(pacman)
    if pacman_position is not None:
        x, y = pacman_position
        if 0 <= y < height and 0 <= x < width:
            observation[3, y, x] = 1.0

    dangerous = []
    frightened = []
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            if entity.get("role") != "ghost" and entity.get("id") not in {"blinky", "pinky", "inky", "clyde"}:
                continue
            position = _position_from_entity(entity)
            if position is None:
                continue
            if entity.get("state") == "frightened" or entity.get("frightened"):
                frightened.append(position)
            else:
                dangerous.append(position)
    else:
        ghosts = state.get("ghosts") or []
        for ghost in ghosts:
            if isinstance(ghost, Mapping):
                position = _position_from_entity(ghost)
                if position is not None:
                    if ghost.get("state") == "frightened":
                        frightened.append(position)
                    else:
                        dangerous.append(position)

    for x, y in dangerous:
        if 0 <= y < height and 0 <= x < width:
            observation[4, y, x] = 1.0
    for x, y in frightened:
        if 0 <= y < height and 0 <= x < width:
            observation[5, y, x] = 1.0

    return observation
