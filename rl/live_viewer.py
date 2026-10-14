from __future__ import annotations

import argparse
import copy
import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from rl.agents import DQNAgent
from rl.environment import PacmanEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = Path(
    "runs/learning-seed73-20261014/"
    "pacman-dqn-20261014T133045092713Z-seed73-2ae528fa/checkpoint.pt"
)
OBSERVATION_SHAPE = (6, 31, 28)
ACTION_COUNT = 4
MIN_INTERVAL_MS = 100
MAX_INTERVAL_MS = 3000


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    requested = Path(checkpoint)
    resolved = (PROJECT_ROOT / requested).resolve() if not requested.is_absolute() else requested.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Checkpoint must be located inside the project directory.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {resolved}")
    return resolved


class PlaybackController:
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        game_server_url: str = "http://127.0.0.1:3000",
        interval_ms: int = 500,
        max_steps: int = 200,
        env_factory: Callable[..., PacmanEnv] | None = None,
        agent_loader: Callable[..., DQNAgent] | None = None,
    ) -> None:
        if not MIN_INTERVAL_MS <= interval_ms <= MAX_INTERVAL_MS:
            raise ValueError(
                f"Playback interval must be between {MIN_INTERVAL_MS} and {MAX_INTERVAL_MS} ms."
            )
        self.checkpoint = checkpoint
        self.game_server_url = game_server_url
        self.max_steps = max_steps
        self._interval_ms = interval_ms
        self._env_factory = env_factory or PacmanEnv
        self._agent_loader = agent_loader or DQNAgent.load_checkpoint
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._env: PacmanEnv | None = None
        self._owner_id: str | None = None
        self._stop_requested = False
        self._paused = False
        self._status = "stopped"
        self._error: str | None = None
        self._steps = 0
        self._snapshot: dict[str, Any] = self._empty_snapshot()

    def _empty_snapshot(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "episode_status": "not_started",
            "checkpoint": str(self.checkpoint),
            "observation_shape": list(OBSERVATION_SHAPE),
            "action_count": ACTION_COUNT,
            "speed_ms": self._interval_ms,
            "steps": 0,
            "current_action": None,
            "game_score": 0,
            "pellets_collected": 0,
            "pellets_remaining": 0,
            "pellets_total": 0,
            "lives": None,
            "step_reward": 0.0,
            "episode_return": 0.0,
            "game_state": None,
            "error": None,
        }

    def state(self) -> dict[str, Any]:
        with self._condition:
            result = copy.deepcopy(self._snapshot)
            result["status"] = self._status
            result["speed_ms"] = self._interval_ms
            result["error"] = self._error
            return result

    def start(self) -> dict[str, Any]:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Playback is already active.")
            self._status = "loading"
            self._error = None

        env: PacmanEnv | None = None
        owner_id = uuid.uuid4().hex
        claimed = False
        try:
            checkpoint_path = resolve_checkpoint(self.checkpoint)
            agent = self._agent_loader(
                checkpoint_path,
                OBSERVATION_SHAPE,
                ACTION_COUNT,
                device="cpu",
            )
            agent.online_network.eval()
            env = self._env_factory(
                base_url=self.game_server_url,
                timeout=5.0,
                max_steps=self.max_steps,
                owner_id=owner_id,
            )
            env.game_client.claim(owner_id)
            claimed = True
            observation, info = env.reset()
            maze = env.game_client.get_maze()
            maze_rows = maze.get("rows")
            if not isinstance(maze_rows, list) or not maze_rows:
                raise ValueError("Game server did not provide canonical maze rows.")
            initial_state = self._viewer_state(info["state"], maze_rows)
        except Exception as exc:
            cleanup_error = self._release_resources(env, owner_id, claimed)
            error = f"{exc}"
            if cleanup_error:
                error = f"{error}; cleanup failed: {cleanup_error}"
            with self._condition:
                self._status = "error"
                self._error = error
                self._snapshot = self._empty_snapshot()
                self._snapshot["status"] = "error"
                self._snapshot["error"] = error
            raise RuntimeError(error) from exc

        with self._condition:
            self._stop_requested = False
            self._paused = False
            self._steps = 0
            self._owner_id = owner_id
            self._env = env
            self._status = "playing"
            self._snapshot = self._make_snapshot(
                state=initial_state,
                info=info,
                action=None,
                step_reward=0.0,
                steps=0,
                episode_status="playing",
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(env, agent, observation, owner_id),
                name="pacman-dqn-playback",
                daemon=True,
            )
            self._thread.start()
            return copy.deepcopy(self._snapshot)

    def pause(self) -> dict[str, Any]:
        with self._condition:
            if self._status != "playing":
                raise RuntimeError("Playback is not currently playing.")
            self._paused = True
            self._status = "paused"
            self._condition.notify_all()
            return self.state()

    def resume(self) -> dict[str, Any]:
        with self._condition:
            if self._status != "paused":
                raise RuntimeError("Playback is not paused.")
            self._paused = False
            self._status = "playing"
            self._condition.notify_all()
            return copy.deepcopy(self._snapshot)

    def stop(self) -> dict[str, Any]:
        with self._condition:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._status = "stopped"
                self._paused = False
                self._snapshot["status"] = "stopped"
                return copy.deepcopy(self._snapshot)
            self._stop_requested = True
            self._status = "stopping"
            self._condition.notify_all()

        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("Playback did not stop within 10 seconds.")
        return self.state()

    def set_speed(self, interval_ms: int) -> dict[str, Any]:
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
            raise ValueError("Playback speed interval must be an integer in milliseconds.")
        if not MIN_INTERVAL_MS <= interval_ms <= MAX_INTERVAL_MS:
            raise ValueError(
                f"Playback interval must be between {MIN_INTERVAL_MS} and {MAX_INTERVAL_MS} ms."
            )
        with self._condition:
            self._interval_ms = interval_ms
            self._snapshot["speed_ms"] = interval_ms
            self._condition.notify_all()
            return copy.deepcopy(self._snapshot)

    @staticmethod
    def _viewer_state(state: Mapping[str, Any], maze_rows: list[str]) -> dict[str, Any]:
        result = copy.deepcopy(dict(state))
        maze = result.get("maze")
        result["maze"] = {
            **(dict(maze) if isinstance(maze, Mapping) else {}),
            "rows": maze_rows,
        }
        return result

    def _make_snapshot(
        self,
        *,
        state: Mapping[str, Any],
        info: Mapping[str, Any],
        action: int | None,
        step_reward: float,
        steps: int,
        episode_status: str,
    ) -> dict[str, Any]:
        pellets_total = int(state.get("pelletsTotal", 0))
        pellets_remaining = int(state.get("pelletsRemaining", 0))
        return {
            "status": self._status,
            "episode_status": episode_status,
            "checkpoint": str(self.checkpoint),
            "observation_shape": list(OBSERVATION_SHAPE),
            "action_count": ACTION_COUNT,
            "speed_ms": self._interval_ms,
            "steps": steps,
            "current_action": (
                {"id": action, "name": info.get("action_name")}
                if action is not None
                else None
            ),
            "game_score": float(info.get("game_score", state.get("score", 0.0))),
            "pellets_collected": max(0, pellets_total - pellets_remaining),
            "pellets_remaining": pellets_remaining,
            "pellets_total": pellets_total,
            "lives": state.get("lives"),
            "step_reward": float(step_reward),
            "episode_return": float(info.get("episode_return", 0.0)),
            "game_state": dict(state),
            "error": self._error,
        }

    @staticmethod
    def _episode_status(terminated: bool, truncated: bool, state: Mapping[str, Any]) -> str:
        if bool(state.get("won")):
            return "won"
        if bool(state.get("gameOver")):
            return "lost"
        if truncated:
            return "truncated"
        if terminated:
            return "ended"
        return "playing"

    def _run(
        self,
        env: PacmanEnv,
        agent: DQNAgent,
        observation: Any,
        owner_id: str,
    ) -> None:
        try:
            maze_rows = self._snapshot["game_state"]["maze"]["rows"]
            while True:
                with self._condition:
                    while self._paused and not self._stop_requested:
                        self._condition.wait()
                    if self._stop_requested:
                        break
                    self._condition.wait(timeout=self._interval_ms / 1000.0)
                    if self._stop_requested:
                        break
                    if self._paused:
                        continue

                    action = agent.select_action(observation, explore=False)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    state = self._viewer_state(info["state"], maze_rows)
                    self._steps += 1
                    episode_status = self._episode_status(terminated, truncated, state)
                    self._snapshot = self._make_snapshot(
                        state=state,
                        info=info,
                        action=action,
                        step_reward=reward,
                        steps=self._steps,
                        episode_status=episode_status,
                    )
                    if terminated or truncated:
                        self._status = episode_status
                        self._condition.notify_all()
                        break
                    observation = next_observation
        except Exception as exc:
            with self._condition:
                self._status = "error"
                self._error = str(exc)
                self._snapshot["status"] = "error"
                self._snapshot["error"] = self._error
                self._condition.notify_all()
        finally:
            cleanup_error = self._release_resources(env, owner_id, True)
            with self._condition:
                self._env = None
                self._owner_id = None
                if cleanup_error:
                    self._status = "error"
                    self._error = f"{self._error + '; ' if self._error else ''}cleanup failed: {cleanup_error}"
                    self._snapshot["error"] = self._error
                elif self._stop_requested:
                    self._status = "stopped"
                self._snapshot["status"] = self._status
                self._condition.notify_all()

    @staticmethod
    def _release_resources(env: PacmanEnv | None, owner_id: str, claimed: bool) -> str | None:
        if env is None:
            return None
        errors: list[str] = []
        if claimed:
            try:
                env.game_client.release()
            except Exception as exc:
                errors.append(f"could not release game ownership: {exc}")
        try:
            env.close()
        except Exception as exc:
            errors.append(f"could not close the game environment: {exc}")
        return "; ".join(errors) or None


class ViewerRequestHandler(BaseHTTPRequestHandler):
    controller: PlaybackController
    public_dir = PROJECT_ROOT / "public"
    static_files = {
        "/viewer": ("viewer.html", "text/html; charset=utf-8"),
        "/viewer.html": ("viewer.html", "text/html; charset=utf-8"),
        "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
        "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    }

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 16_384:
            raise ValueError("Request payload is too large.")
        if length == 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/playback/state":
            self._send_json(HTTPStatus.OK, self.controller.state())
            return
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "pacman-live-viewer"})
            return
        static = self.static_files.get(path)
        if static is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        filename, content_type = static
        body = (self.public_dir / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/playback/start":
                result = self.controller.start()
            elif path == "/api/playback/pause":
                result = self.controller.pause()
            elif path == "/api/playback/resume":
                result = self.controller.resume()
            elif path == "/api/playback/stop":
                result = self.controller.stop()
            elif path == "/api/playback/speed":
                payload = self._read_json()
                result = self.controller.set_speed(payload.get("interval_ms"))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            self._send_json(HTTPStatus.OK, result)
        except (ValueError, FileNotFoundError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[viewer] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve live playback for a trained Pac-Man DQN.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--game-server", default="http://127.0.0.1:3000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--speed-ms", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    controller = PlaybackController(
        checkpoint=args.checkpoint,
        game_server_url=args.game_server,
        interval_ms=args.speed_ms,
        max_steps=args.max_steps,
    )
    handler = type("BoundViewerRequestHandler", (ViewerRequestHandler,), {"controller": controller})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    print(f"Live DQN viewer: http://{args.host}:{args.port}/viewer")
    print(f"Checkpoint (validated on Start): {args.checkpoint}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live viewer...")
    finally:
        controller.stop()
        server.server_close()


if __name__ == "__main__":
    main()
