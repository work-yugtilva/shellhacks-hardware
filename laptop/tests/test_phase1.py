import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parents[1]))

from lanetalk.camera import CameraStream
from lanetalk.events import Event


class _UnavailableCapture:
    def open(self, *_args):
        return False

    def release(self):
        pass


class _FakeResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks

    def close(self):
        self.closed = True


class PhaseOneTests(unittest.TestCase):
    def test_http_fallback_decodes_multipart_jpeg_to_bgr(self):
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        source[:, :] = (10, 80, 200)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        chunks = [body[:37], body[37:113], body[113:]]
        response = _FakeResponse(chunks)

        with patch("lanetalk.camera.cv2.VideoCapture", return_value=_UnavailableCapture()):
            with patch("lanetalk.camera.requests.get", return_value=response):
                camera = CameraStream(reconnect_delay=0)
                try:
                    frame = camera.read()
                finally:
                    camera.close()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (12, 16, 3))
        self.assertGreater(float(frame[:, :, 2].mean()), float(frame[:, :, 0].mean()))

    def test_reconnect_count_tracks_successful_reconnects_only(self):
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)
        body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        initial_response = _FakeResponse([body])
        reconnected_response = _FakeResponse([body])

        with patch("lanetalk.camera.cv2.VideoCapture", return_value=_UnavailableCapture()):
            with patch(
                "lanetalk.camera.requests.get",
                side_effect=[
                    initial_response,
                    requests.ConnectionError("offline"),
                    reconnected_response,
                ],
            ):
                camera = CameraStream(reconnect_delay=0)
                try:
                    self.assertIsNotNone(camera.read())
                    self.assertEqual(camera.reconnect_count, 0)

                    self.assertIsNone(camera.read())
                    self.assertEqual(camera.reconnect_count, 0)

                    self.assertIsNone(camera.read())
                    self.assertEqual(camera.reconnect_count, 0)

                    self.assertIsNotNone(camera.read())
                    self.assertEqual(camera.reconnect_count, 1)
                finally:
                    camera.close()

    def test_event_has_epoch_timestamp_and_independent_details(self):
        first = Event(kind="test", confidence=0.5)
        second = Event(kind="test", confidence=0.5)

        self.assertIsInstance(first.timestamp, float)
        self.assertGreater(first.timestamp, 0)
        self.assertIsNot(first.details, second.details)


if __name__ == "__main__":
    unittest.main()
