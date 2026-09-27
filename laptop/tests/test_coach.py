import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lanetalk import coach, speech
from lanetalk.events import Event

LEFT_DRIFT = "Drifting left. Steer gently back to center."


class CoachLineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {"LANETALK_TIPS_DB": str(Path(tmp.name) / "tips.db")})
        env.start()
        self.addCleanup(env.stop)
        self.event = Event("lane_drift", 0.9, details={"direction": "left", "offset": -0.3})

    def _reply(self, text):
        response = mock.Mock()
        response.json.return_value = {"response": text}
        return response

    def test_uses_first_line_of_model_reply_and_sends_tips(self):
        with mock.patch.object(coach.requests, "post", return_value=self._reply("Ease right now.\nextra")) as post:
            self.assertEqual(coach.coach_line(self.event), "Ease right now.")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], coach.get_ollama_model())
        self.assertIn(LEFT_DRIFT, body["prompt"])
        self.assertFalse(body["stream"])

    def test_falls_back_to_first_tip_when_ollama_down_or_empty(self):
        with mock.patch.object(coach.requests, "post", side_effect=requests.ConnectionError("down")):
            self.assertEqual(coach.coach_line(self.event), LEFT_DRIFT)
        with mock.patch.object(coach.requests, "post", side_effect=requests.Timeout("slow")):
            self.assertEqual(coach.coach_line(self.event), LEFT_DRIFT)
        with mock.patch.object(coach.requests, "post", return_value=self._reply("  ")):
            self.assertEqual(coach.coach_line(self.event), LEFT_DRIFT)

    def test_configurable_model_tag(self):
        with mock.patch.dict(os.environ, {"OLLAMA_MODEL": "custom-gemma4:tag"}):
            with mock.patch.object(coach.requests, "post", return_value=self._reply("Hold line.")) as post:
                self.assertEqual(coach.coach_line(self.event), "Hold line.")
            body = post.call_args.kwargs["json"]
            self.assertEqual(body["model"], "custom-gemma4:tag")

    def test_unknown_event_has_no_line(self):
        with mock.patch.object(coach.requests, "post") as post:
            self.assertEqual(coach.coach_line(Event("unknown", 0.5)), "")
        post.assert_not_called()


class CoachWorkerTests(unittest.TestCase):
    def test_per_kind_cooldown(self):
        worker = coach.CoachWorker(cooldown_seconds=60, line_fn=lambda e: "", say_fn=lambda t: True)
        self.addCleanup(worker.close)
        self.assertTrue(worker.submit(Event("lane_drift", 0.9)))
        self.assertFalse(worker.submit(Event("lane_drift", 0.9)))
        self.assertTrue(worker.submit(Event("vehicle_detected", 0.9)))

    def test_keeps_only_latest_pending_event_and_drops_stale(self):
        release = threading.Event()
        spoken = []
        done = threading.Event()

        def line_fn(event):
            if event.kind == "first":
                release.wait(2)
            return event.kind

        def say_fn(text):
            spoken.append(text)
            if text == "latest":
                done.set()
            return True

        worker = coach.CoachWorker(cooldown_seconds=0, line_fn=line_fn, say_fn=say_fn)
        self.addCleanup(worker.close)
        worker.submit(Event("first", 0.9))
        while not worker._pending.empty():
            pass
        worker.submit(Event("old", 0.9, timestamp=0.0))
        worker.submit(Event("middle", 0.9))
        worker.submit(Event("latest", 0.9))
        release.set()
        self.assertTrue(done.wait(2))
        self.assertEqual(spoken, ["first", "latest"])

    def test_stale_event_is_skipped(self):
        line_fn = mock.Mock(return_value="x")
        worker = coach.CoachWorker(line_fn=line_fn, say_fn=lambda t: True)
        worker.submit(Event("lane_drift", 0.9, timestamp=0.0))
        while not worker._pending.empty():
            pass
        worker.close()
        line_fn.assert_not_called()

    def test_fish_speech_timeout_and_fallback_run_off_frame_loop(self):
        started_fish_speech = threading.Event()
        fallback_invoked = threading.Event()

        def custom_post(url, *args, **kwargs):
            if "11434" in url or "generate" in url:
                raise requests.ConnectionError("No Ollama")
            if "8080" in url or "tts" in url:
                started_fish_speech.set()
                time.sleep(0.05)
                raise requests.Timeout("Fish Speech connection timed out")
            return mock.MagicMock()

        def mock_fallback(text):
            fallback_invoked.set()
            return True

        speech._next_server_retry = 0.0
        speech._last_error_message = ""

        worker = coach.CoachWorker(cooldown_seconds=0, say_fn=speech.say)
        self.addCleanup(worker.close)

        with mock.patch("requests.post", side_effect=custom_post), \
             mock.patch("lanetalk.speech._speak_fallback", side_effect=mock_fallback):
            event = Event("lane_drift", 0.9, details={"direction": "left"})

            frame_start = time.perf_counter()
            accepted = worker.submit(event)
            submit_elapsed_ms = (time.perf_counter() - frame_start) * 1000

            self.assertTrue(accepted)
            self.assertLess(submit_elapsed_ms, 5.0)

            self.assertTrue(started_fish_speech.wait(timeout=2.0))
            self.assertTrue(fallback_invoked.wait(timeout=2.0))


if __name__ == "__main__":
    unittest.main()
