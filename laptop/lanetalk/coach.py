"""Turn detector events into short spoken coaching lines.

Tips come from the local SQLite retrieval module; local Gemma 4 via Ollama
rewrites them into one line. If Ollama is unavailable, times out, or returns
nothing, the first retrieved tip is used instead.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable

import requests

from lanetalk import speech
from lanetalk.events import Event
from lanetalk.retrieval import retrieve

DEFAULT_OLLAMA_MODEL = "gemma4"
OLLAMA_TIMEOUT_SECONDS = 4.0


def get_ollama_model() -> str:
    """Return the configured Ollama model tag."""
    return (
        os.environ.get("OLLAMA_MODEL")
        or os.environ.get("LANETALK_OLLAMA_MODEL")
        or os.environ.get("LANETALK_MODEL")
        or DEFAULT_OLLAMA_MODEL
    ).strip() or DEFAULT_OLLAMA_MODEL


def get_ollama_endpoint() -> str:
    """Return the full Ollama API endpoint."""
    url = (
        os.environ.get("OLLAMA_URL")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    if url.endswith("/api/generate") or url.endswith("/api/chat"):
        return url
    return f"{url}/api/generate"


# Module-level variable for tests and inspectability
OLLAMA_MODEL = get_ollama_model()


def _prompt(event: Event, tips: list[str]) -> str:
    details = {key: value for key, value in event.details.items() if key != "bbox"}
    tip_lines = "\n".join(f"- {tip}" for tip in tips)
    return (
        "You are a calm in-car driving coach. Reply with ONE short spoken sentence "
        "(under 12 words), no preamble, no quotes.\n"
        f"Event: {event.kind} {details}\n"
        f"Reference tips:\n{tip_lines}"
    )


def coach_line(event: Event, timeout: float = OLLAMA_TIMEOUT_SECONDS) -> str:
    """Return a brief coaching line, falling back to the first retrieved tip."""
    tips = retrieve(event, top_k=3)
    if not tips:
        return ""

    model = get_ollama_model()
    if "cloud" in model.casefold():
        print("[coach] Cloud Ollama models are disabled; using retrieved tip.", flush=True)
        return tips[0]

    endpoint = get_ollama_endpoint()
    prompt_text = _prompt(event, tips)

    try:
        if endpoint.endswith("/api/chat"):
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a calm in-car driving coach. Use the event and reference tips "
                            "to give one short spoken sentence of at most 12 words. No preamble or quotes."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 40},
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 40},
            }

        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
            proxies={"http": "", "https": ""},
        )
        response.raise_for_status()
        data = response.json()
        raw_text = ""
        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], str):
                raw_text = data["response"]
            elif "message" in data and isinstance(data["message"], dict):
                raw_text = data["message"].get("content", "")

        lines = raw_text.strip().splitlines() if raw_text else []
        text = lines[0].strip().strip('"').strip() if lines else ""
    except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
        print(f"[coach] Ollama unavailable, using tip: {exc}", flush=True)
        text = ""

    return text or tips[0]


class CoachWorker:
    """Generate and speak coaching lines for events off the camera frame loop."""

    def __init__(
        self,
        cooldown_seconds: float = 6.0,
        max_age_seconds: float = 3.0,
        line_fn: Callable[[Event], str] = coach_line,
        say_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.max_age_seconds = max_age_seconds
        self._line_fn = line_fn
        self._say_fn = say_fn or speech.say
        self._pending: queue.Queue[Event | None] = queue.Queue(maxsize=1)
        self._last_accepted: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, name="lanetalk-coach", daemon=True)
        self._thread.start()

    def submit(self, event: Event) -> bool:
        """Queue one emitted event without waiting for local inference or TTS."""
        if self.cooldown_seconds > 0:
            now = time.monotonic()
            if now - self._last_accepted.get(event.kind, float("-inf")) < self.cooldown_seconds:
                return False
            self._last_accepted[event.kind] = now
        self._replace_pending(event)
        return True

    def close(self) -> None:
        """Drain queued events before stopping the worker."""
        self._replace_pending(None)
        self._thread.join(timeout=2.0)

    def _replace_pending(self, item: Event | None) -> None:
        try:
            self._pending.get_nowait()
        except queue.Empty:
            pass
        self._pending.put_nowait(item)

    def _run(self) -> None:
        while True:
            event = self._pending.get()
            if event is None:
                return
            if self.max_age_seconds > 0 and (time.time() - event.timestamp > self.max_age_seconds):
                continue
            try:
                text = self._line_fn(event)
            except Exception as exc:
                print(f"[coach] Event coaching failed: {exc}", flush=True)
                continue
            if text:
                print(f"[coach] {text}", flush=True)
                try:
                    self._say_fn(text)
                except Exception as exc:
                    print(f"[coach] Speech failed: {exc}", flush=True)
