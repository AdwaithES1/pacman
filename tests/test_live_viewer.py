from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests

from rl.environment.pacman_env import PacmanEnv
from rl.live_viewer import (
    DEFAULT_CHECKPOINT,
    PROJECT_ROOT,
    PlaybackController,
    resolve_checkpoint,
)


MAZE_ROWS = ["#" * 28 for _ in range(31)]


class FakeGameClient:
    def __init__(self) -> None:
        self.claimed: str | None = None
        self.released = False

    def claim(self, owner_id: str) -> None:
        self.claimed = owner_id

    def release(self) -> None:
        self.released = True

    def get_maze(self) -> dict[str, Any]:
        return {"rows": MAZE_ROWS}


class FakeEnv:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        max_steps: int,
        owner_id: str,
        terminal: bool = False,
        truncated: bool = False,
    ) -> None:
        self.game_client = FakeGameClient()
        self.owner_id = owner_id
        self.terminal = terminal
        self.truncated = truncated
        self.closed = False
        self.step_event = threading.Event()
        self.actions: list[int] = []
        self.state: dict[str, Any] = {}

    def reset(self):
        self.state = {
            "score": 0,
            "lives": 3,
            "pelletsTotal": 10,
            "pelletsRemaining": 10,
            "stepCount": 0,
            "won": False,
            "gameOver": False,
            "maze": {"width": 28, "height": 31},
            "entities": [],
            "pelletRows": [" " * 28 for _ in range(31)],
        }
        return np.zeros((6, 31, 28), dtype=np.float32), {
            "state": self.state,
            "game_score": 0,
            "episode_return": 0.0,
        }

    def step(self, action: int):
        self.actions.append(action)
        self.state = {
            **self.state,
            "stepCount": len(self.actions),
            "score": len(self.actions),
            "pelletsRemaining": 10 - len(self.actions),
            "won": self.terminal,
            "gameOver": False,
        }
        self.step_event.set()
        action_name = PacmanEnv.ACTION_MAP[action]
        info = {
            "state": self.state,
            "action_name": action_name,
            "game_score": self.state["score"],
            "episode_return": float(len(self.actions)),
        }
        return (
            np.zeros((6, 31, 28), dtype=np.float32),
            1.0,
            self.terminal,
            self.truncated,
            info,
        )

    def close(self) -> None:
        self.closed = True


class FakeNetwork:
    def __init__(self) -> None:
        self.evaluating = False

    def eval(self) -> None:
        self.evaluating = True


class FakeAgent:
    def __init__(self) -> None:
        self.online_network = FakeNetwork()
        self.exploration_flags: list[bool] = []

    def select_action(self, observation: np.ndarray, *, explore: bool) -> int:
        self.exploration_flags.append(explore)
        return 3


def make_controller(
    monkeypatch: pytest.MonkeyPatch,
    envs: list[FakeEnv],
    *,
    terminal: bool = False,
    truncated: bool = False,
) -> tuple[PlaybackController, FakeAgent]:
    agent = FakeAgent()

    def env_factory(**kwargs: Any) -> FakeEnv:
        env = FakeEnv(**kwargs, terminal=terminal, truncated=truncated)
        envs.append(env)
        return env

    monkeypatch.setattr("rl.live_viewer.resolve_checkpoint", lambda checkpoint: Path(__file__))
    controller = PlaybackController(
        checkpoint=DEFAULT_CHECKPOINT,
        interval_ms=100,
        max_steps=2,
        env_factory=env_factory,
        agent_loader=lambda *args, **kwargs: agent,
    )
    return controller, agent


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_checkpoint_path_is_confined_to_project_and_exists():
    checkpoint = resolve_checkpoint(DEFAULT_CHECKPOINT)
    assert checkpoint.is_relative_to(PROJECT_ROOT)
    assert checkpoint.is_file()
    with pytest.raises(ValueError, match="inside the project"):
        resolve_checkpoint(PROJECT_ROOT.parent / "outside.pt")


def test_start_pause_resume_speed_and_stop_release_resources(monkeypatch):
    envs: list[FakeEnv] = []
    controller, agent = make_controller(monkeypatch, envs)

    started = controller.start()
    assert started["status"] == "playing"
    assert started["checkpoint"].endswith(str(DEFAULT_CHECKPOINT))
    assert agent.online_network.evaluating

    paused = controller.pause()
    assert paused["status"] == "paused"
    time.sleep(0.15)
    assert envs[0].actions == []

    resumed = controller.resume()
    assert resumed["status"] == "playing"
    assert envs[0].step_event.wait(timeout=2)
    assert agent.exploration_flags == [False]

    speed = controller.set_speed(200)
    assert speed["speed_ms"] == 200
    paused = controller.pause()
    assert paused["status"] == "paused"
    stopped = controller.stop()
    action_count = len(envs[0].actions)
    time.sleep(0.25)

    assert stopped["status"] == "stopped"
    assert len(envs[0].actions) == action_count
    assert envs[0].game_client.released
    assert envs[0].closed


@pytest.mark.parametrize(
    ("terminal", "truncated", "expected"),
    [(True, False, "won"), (False, True, "truncated")],
)
def test_episode_end_states_release_environment(
    monkeypatch,
    terminal: bool,
    truncated: bool,
    expected: str,
):
    envs: list[FakeEnv] = []
    controller, _agent = make_controller(
        monkeypatch,
        envs,
        terminal=terminal,
        truncated=truncated,
    )
    controller.start()
    assert wait_for(lambda: controller.state()["status"] == expected)
    state = controller.state()
    assert state["episode_status"] == expected
    assert state["steps"] == 1
    assert envs[0].game_client.released
    assert envs[0].closed


@pytest.fixture
def live_server():
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for live playback integration testing")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    proc = subprocess.Popen(
        [node_executable, "server.js"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PORT": str(port), "HOST": "127.0.0.1"},
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/api/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        proc.terminate()
        pytest.fail("Pac-Man game server failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_real_checkpoint_drives_live_game_server(live_server):
    controller = PlaybackController(
        checkpoint=DEFAULT_CHECKPOINT,
        game_server_url=live_server,
        interval_ms=100,
        max_steps=2,
    )
    try:
        started = controller.start()
        assert started["status"] == "playing"
        assert started["observation_shape"] == [6, 31, 28]
        assert started["action_count"] == 4
        assert started["game_state"]["maze"]["rows"]
        assert wait_for(lambda: controller.state()["status"] == "truncated", timeout=20)
        final = controller.state()
        assert final["steps"] == 2
        assert final["episode_status"] == "truncated"
        assert final["current_action"]["id"] in range(4)
        assert final["current_action"]["name"] == PacmanEnv.ACTION_MAP[final["current_action"]["id"]]
        assert final["game_state"]["stepCount"] == 2
        assert isinstance(final["step_reward"], float)
        assert isinstance(final["episode_return"], float)
    finally:
        controller.stop()

    reset = requests.post(
        f"{live_server}/api/agent/reset",
        json={"max_steps": 2},
        timeout=2,
    )
    assert reset.status_code == 200, reset.text


def test_viewer_http_route_serves_page_and_state(monkeypatch):
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    from rl.live_viewer import ViewerRequestHandler

    envs: list[FakeEnv] = []
    controller, _agent = make_controller(monkeypatch, envs)
    handler = type("TestViewerHandler", (ViewerRequestHandler,), {"controller": controller})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base_url}/viewer") as response:
            assert response.status == 200
            assert b"Live DQN Viewer" in response.read()
        with urlopen(f"{base_url}/api/playback/state") as response:
            assert b'"status": "stopped"' in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
