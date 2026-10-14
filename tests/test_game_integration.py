import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests
import numpy as np

from rl.environment import PacmanEnv


@pytest.fixture(scope="module")
def live_server():
    project_root = Path(__file__).resolve().parent.parent
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for server integration tests")
    proc = subprocess.Popen(
        [node_executable, "server.js"],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PORT": "3000", "HOST": "127.0.0.1"},
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            response = requests.get("http://127.0.0.1:3000/api/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail("Pac-Man server failed to start for integration testing")
    try:
        yield "http://127.0.0.1:3000"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_server_is_reachable(live_server):
    response = requests.get(f"{live_server}/api/health", timeout=2)
    assert response.status_code == 200


def test_reset_and_step_endpoints_work(live_server):
    reset = requests.post(f"{live_server}/api/agent/reset", json={"max_steps": 10}, timeout=2)
    assert reset.status_code == 200, reset.text
    step = requests.post(f"{live_server}/api/agent/step", json={"action": 0}, timeout=2)
    assert step.status_code == 200, step.text


def test_environment_observation_contains_canonical_walls_and_live_entities(live_server):
    env = PacmanEnv(base_url=live_server, timeout=2.0)
    try:
        observation, _ = env.reset(seed=11)
        assert observation.shape == (6, 31, 28)
        assert observation[0].sum() > 0
        assert observation[1].sum() + observation[2].sum() > 0
        assert observation[3].sum() == 1
        assert observation[4].sum() + observation[5].sum() == 4

        next_observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert info["action_name"] == "UP"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
