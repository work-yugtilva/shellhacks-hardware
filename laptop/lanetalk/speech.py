"""Speech synthesis module for LaneTalk driving coaching.

Integrates with a locally running Fish Speech HTTP service (defaulting to
http://127.0.0.1:8080/v1/tts) for neural spoken coaching without cloud API
dependencies or API keys. If the local service is unreachable or encounters
an error, gracefully falls back to local platform text-to-speech (macOS `say`,
Windows SAPI.SpVoice, Linux `spd-say`/`espeak`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import requests

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_ENDPOINT_PATH = "/v1/tts"
DEFAULT_TIMEOUT_SECONDS = 5.0
RETRY_COOLDOWN_SECONDS = 15.0

_next_server_retry: float = 0.0
_last_error_message: str = ""


def get_service_url() -> str:
    """Resolve the full URL for the Fish Speech TTS endpoint."""
    url = (
        os.environ.get("FISH_SPEECH_URL")
        or os.environ.get("FISH_SPEECH_ENDPOINT")
        or DEFAULT_SERVER_URL
    ).strip()
    if url.endswith("/v1/tts"):
        return url
    return url.rstrip("/") + DEFAULT_ENDPOINT_PATH


def _synthesize_fish_speech(text: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes | None:
    """Send synthesis request to local Fish Speech service and return WAV bytes."""
    global _next_server_retry, _last_error_message

    if os.environ.get("FISH_SPEECH_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return None

    if time.monotonic() < _next_server_retry:
        return None

    url = get_service_url()
    payload: dict[str, Any] = {
        "text": text,
        "format": "wav",
    }
    reference_id = os.environ.get("FISH_SPEECH_REFERENCE_ID")
    if reference_id:
        payload["reference_id"] = reference_id

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.content
        if not content:
            raise ValueError("Empty audio payload received from Fish Speech service")
        _next_server_retry = 0.0
        _last_error_message = ""
        return content
    except Exception as exc:
        _next_server_retry = time.monotonic() + RETRY_COOLDOWN_SECONDS
        err_str = f"{type(exc).__name__}: {exc}"
        if err_str != _last_error_message:
            print(
                f"[speech] Fish Speech unavailable ({err_str}); using local speech fallback.",
                flush=True,
            )
            _last_error_message = err_str
        return None


def _play_audio_bytes(audio_bytes: bytes) -> bool:
    """Play WAV audio bytes using platform audio utilities."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # macOS: afplay
        if sys.platform == "darwin" and shutil.which("afplay"):
            subprocess.run(["afplay", tmp_path], check=True, timeout=30)
            return True

        # Windows: winsound or PowerShell SoundPlayer
        if os.name == "nt":
            try:
                import winsound

                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                return True
            except Exception:
                windir = os.environ.get("WINDIR", r"C:\Windows")
                powershell = os.path.join(
                    windir, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
                )
                cmd = f'(New-Object Media.SoundPlayer "{tmp_path}").PlaySync()'
                subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command", cmd],
                    check=True,
                    timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                return True

        # Linux / Unix: check available CLI audio players
        for player in ("aplay", "paplay", "pw-play", "ffplay", "mpv"):
            player_path = shutil.which(player)
            if player_path:
                if player == "ffplay":
                    cmd = [player_path, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
                elif player == "mpv":
                    cmd = [player_path, "--no-video", tmp_path]
                else:
                    cmd = [player_path, tmp_path]
                subprocess.run(cmd, check=True, timeout=30)
                return True

        return False
    except Exception as exc:
        print(f"[speech] Audio playback failed ({type(exc).__name__}: {exc})", flush=True)
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _speak_windows(text: str) -> bool:
    """Windows SAPI.SpVoice TTS fallback."""
    try:
        windir = os.environ.get("WINDIR", r"C:\Windows")
        powershell = os.path.join(
            windir, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
        )
        command = (
            '$ErrorActionPreference = "Stop"; '
            "$voice = New-Object -ComObject SAPI.SpVoice; "
            "[void]$voice.Speak([Console]::In.ReadToEnd())"
        )
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            input=text,
            text=True,
            capture_output=True,
            check=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except Exception:
        return False


def _speak_macos(text: str) -> bool:
    """macOS `say` TTS fallback."""
    if shutil.which("say"):
        try:
            subprocess.run(["say", text], check=True, timeout=15)
            return True
        except Exception:
            return False
    return False


def _speak_linux(text: str) -> bool:
    """Linux `spd-say` or `espeak` TTS fallback."""
    for cmd in ("spd-say", "espeak"):
        if shutil.which(cmd):
            try:
                subprocess.run([cmd, text], check=True, timeout=15)
                return True
            except Exception:
                continue
    return False


def _speak_fallback(text: str) -> bool:
    """Dispatch to platform-specific local TTS fallback."""
    if sys.platform == "darwin":
        if _speak_macos(text):
            return True
    elif os.name == "nt":
        if _speak_windows(text):
            return True
    else:
        if _speak_linux(text):
            return True
    return False


def say(text: str) -> bool:
    """Speak coaching guidance to the driver.

    Attempts neural speech synthesis using the local Fish Speech HTTP service.
    If the service is not running or encounters an error, gracefully falls back
    to the operating system's native local speech engine.

    Parameters
    ----------
    text : str
        The driving tip or coaching instruction to speak.

    Returns
    -------
    bool
        True if the text was spoken (either via Fish Speech or fallback),
        or False if speech output could not be played.
    """
    if not text or not text.strip():
        return False

    clean_text = text.strip()
    print(f"LaneTalk: {clean_text}", flush=True)

    # 1. Attempt Fish Speech local service
    audio_bytes = _synthesize_fish_speech(clean_text)
    if audio_bytes is not None:
        if _play_audio_bytes(audio_bytes):
            return True

    # 2. Local speech engine fallback
    return _speak_fallback(clean_text)


speak = say


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "LaneTalk is online."
    say(message)
