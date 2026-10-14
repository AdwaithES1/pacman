from __future__ import annotations

import os
from typing import Any, Mapping

import requests


class PacmanGameClient:
    """Small HTTP client for the local Pac-Man training API."""

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        self.base_url = (base_url or os.getenv("PACMAN_BASE_URL", "http://127.0.0.1:3000")).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network failures intentionally surfaced
            raise RuntimeError(f"Unable to reach Pac-Man game server at {url}: {exc}") from exc

        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = {"error": response.text}
            error_text = detail.get("error") if isinstance(detail, dict) else str(detail)
            raise RuntimeError(f"Request failed with HTTP {response.status_code}: {error_text}")

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - malformed JSON is a bug in the server
            raise ValueError(f"Server response from {url} was not valid JSON") from exc

    def reset(self, max_steps: int = 200) -> dict[str, Any]:
        payload = {"max_steps": int(max_steps)}
        return self._request("POST", "/api/agent/reset", payload)

    def step(self, action: int) -> dict[str, Any]:
        if not isinstance(action, int):
            raise TypeError("Action must be an integer between 0 and 3.")
        if action < 0 or action > 3:
            raise ValueError("Action must be one of 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.")
        return self._request("POST", "/api/agent/step", {"action": action})

    def get_state(self) -> dict[str, Any]:
        return self._request("GET", "/api/agent/state")

    def get_maze(self) -> dict[str, Any]:
        return self._request("GET", "/api/maze")

    def close(self) -> None:
        self.session.close()


GameClient = PacmanGameClient
