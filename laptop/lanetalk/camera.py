"""OpenCV client for the ESP32-CAM MJPEG stream."""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np
import requests


class CameraStream:
    """Read BGR frames from an MJPEG URL and retry after disconnects."""

    _MAX_JPEG_BYTES = 8 * 1024 * 1024

    def __init__(
        self,
        url: str = "http://192.168.4.1/stream",
        reconnect_delay: float = 1.0,
        timeout_ms: int = 2000,
    ) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.timeout_ms = timeout_ms
        self.last_read_ms = 0.0
        self.last_error = ""
        self._capture: Optional[cv2.VideoCapture] = None
        self._response: Optional[requests.Response] = None
        self._chunks: Optional[Iterator[bytes]] = None
        self._buffer = bytearray()
        self._use_requests = False
        self._retry_at = 0.0
        self._reconnect_count = 0
        self._has_connected = False

    @property
    def reconnect_count(self) -> int:
        """Number of successful reconnects after the initial connection."""
        return self._reconnect_count

    def _close_http(self) -> None:
        if self._response is not None:
            self._response.close()
            self._response = None
        self._chunks = None
        self._buffer.clear()

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._close_http()

    def _open_cv(self) -> bool:
        capture = cv2.VideoCapture()
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.timeout_ms,
        ]
        try:
            opened = capture.open(self.url, cv2.CAP_FFMPEG, params)
        except cv2.error:
            opened = False
        if not opened:
            capture.release()
            return False
        self._capture = capture
        return True

    def _open_requests(self) -> bool:
        response = None
        try:
            response = requests.get(
                self.url,
                stream=True,
                timeout=(self.timeout_ms / 1000, self.timeout_ms / 1000),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            if response is not None:
                response.close()
            self.last_error = f"HTTP stream unavailable: {exc}"
            return False

        self._response = response
        self._chunks = iter(response.iter_content(chunk_size=4096))
        self._buffer.clear()
        self.last_error = ""
        return True

    def _connect(self) -> bool:
        self._release()
        opened = not self._use_requests and self._open_cv()
        if not opened:
            self._use_requests = True
            opened = self._open_requests()
        if opened:
            if self._has_connected:
                self._reconnect_count += 1
            else:
                self._has_connected = True
        return opened

    def _next_jpeg(self) -> Optional[bytes]:
        while self._chunks is not None:
            start = self._buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self._buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    jpeg = bytes(self._buffer[start : end + 2])
                    del self._buffer[: end + 2]
                    return jpeg
                if start:
                    del self._buffer[:start]
                if len(self._buffer) > self._MAX_JPEG_BYTES:
                    raise ValueError("MJPEG frame exceeded 8 MiB")
            elif self._buffer:
                # Keep a possible first byte of an SOI marker split across chunks.
                keep = 1 if self._buffer[-1] == 0xFF else 0
                if keep:
                    del self._buffer[:-1]
                else:
                    self._buffer.clear()

            chunk = next(self._chunks)
            if chunk:
                self._buffer.extend(chunk)
        return None

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None while connecting/reconnecting."""
        if self._capture is None and self._response is None:
            if time.monotonic() < self._retry_at:
                return None
            if not self._connect():
                self._retry_at = time.monotonic() + self.reconnect_delay
                return None

        started = time.monotonic()
        try:
            if self._capture is not None:
                ok, frame = self._capture.read()
                if ok and frame is not None:
                    self.last_read_ms = (time.monotonic() - started) * 1000
                    self.last_error = ""
                    return frame
                self._capture.release()
                self._capture = None
                self._use_requests = True
                self.last_error = "OpenCV stream ended; switching to HTTP MJPEG decoder"
                self._retry_at = time.monotonic() + self.reconnect_delay
                return None

            jpeg = self._next_jpeg()
            if jpeg is None:
                raise ConnectionError("MJPEG response ended")
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode an MJPEG frame")
            self.last_read_ms = (time.monotonic() - started) * 1000
            self.last_error = ""
            return frame
        except (cv2.error, requests.RequestException, StopIteration, OSError, ValueError) as exc:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
                self._use_requests = True
            self._close_http()
            self.last_error = f"Camera stream dropped: {exc}"
            self._retry_at = time.monotonic() + self.reconnect_delay
            return None

    def close(self) -> None:
        self._release()

    def __enter__(self) -> CameraStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
