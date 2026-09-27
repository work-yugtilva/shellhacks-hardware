import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from lanetalk.detector import Detection, LaneDetector, ObjectDetector, _select_device


def _lane_frame(left: tuple[tuple[int, int], tuple[int, int]], right: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(frame, left[0], left[1], (255, 255, 255), 6)
    cv2.line(frame, right[0], right[1], (255, 255, 255), 6)
    return frame


def _road_lane_frame(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Synthetic asphalt with high-contrast lane markings and mild texture."""
    frame = np.full((360, 640, 3), 82, dtype=np.uint8)
    # A deterministic low-contrast texture keeps the road distinct from a flat interior.
    for y in range(200, 360, 16):
        cv2.line(frame, (40, y), (600, y), (88, 88, 88), 1)
    cv2.line(frame, left[0], left[1], color, 7)
    cv2.line(frame, right[0], right[1], color, 7)
    return frame


class ObjectDetectorTests(unittest.TestCase):
    def test_device_selection_prefers_mps_and_falls_back_to_cpu(self):
        mps_torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
        )
        cpu_torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        )
        with patch.dict(sys.modules, {"torch": mps_torch}):
            self.assertEqual(_select_device(), "mps")
        with patch.dict(sys.modules, {"torch": cpu_torch}):
            self.assertEqual(_select_device(), "cpu")

    def test_stride_adapts_to_inference_duration(self):
        self.assertEqual(ObjectDetector._stride_for_timing(3, 150, 31), 5)
        self.assertEqual(ObjectDetector._stride_for_timing(3, 70, 31), 3)

    def test_result_parser_keeps_only_configured_coco_classes(self):
        boxes = SimpleNamespace(
            xyxy=np.array([[1, 2, 10, 20], [20, 30, 40, 60], [5, 5, 15, 15]], dtype=float),
            conf=np.array([0.91, 0.88, 0.99]),
            cls=np.array([0, 2, 16]),
        )
        detections = ObjectDetector._detections_from_results([SimpleNamespace(boxes=boxes)])

        self.assertEqual([item.class_name for item in detections], ["person", "car"])
        self.assertEqual(detections[0].bbox, (1, 2, 10, 20))
        self.assertAlmostEqual(detections[1].confidence, 0.88)

    def test_events_map_classes_and_suppress_recent_overlapping_boxes(self):
        detector = ObjectDetector()
        car = Detection("car", 0.88, (10, 10, 110, 110))
        moved_car = Detection("car", 0.91, (15, 15, 115, 115))
        another_car = Detection("car", 0.83, (300, 200, 380, 280))
        person = Detection("person", 0.95, (10, 10, 60, 100))

        first_events = detector._events_for_detections([car, person], 1.0, 100.0)
        repeated_events = detector._events_for_detections([moved_car], 2.0, 101.0)
        separate_events = detector._events_for_detections([another_car], 2.1, 101.1)
        cooldown_expired = detector._events_for_detections([moved_car], 4.1, 103.1)

        self.assertEqual([event.kind for event in first_events], ["vehicle_detected", "pedestrian_detected"])
        self.assertEqual(first_events[0].details, {"class": "car", "bbox": [10, 10, 110, 110]})
        self.assertEqual(repeated_events, [])
        self.assertEqual(len(separate_events), 1)
        self.assertEqual(len(cooldown_expired), 1)

    def test_worker_uses_cadence_and_keeps_only_latest_pending_frame(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()

        class BlockingModel:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    first_started.set()
                    release_first.wait(timeout=2)
                elif len(self.calls) == 2:
                    second_started.set()
                return [SimpleNamespace(boxes=None)]

        model = BlockingModel()
        with patch("lanetalk.detector._load_yolo_model", return_value=model):
            with patch("lanetalk.detector._select_device", return_value="mps"):
                with patch("lanetalk.detector._warmup_model"):
                    detector = ObjectDetector()
                    detector.start()
                    self.assertTrue(detector.wait_until_ready(2))
                    self.assertFalse(detector.submit(np.zeros((20, 20, 3), dtype=np.uint8), 1, 31))
                    self.assertFalse(detector.submit(np.zeros((20, 20, 3), dtype=np.uint8), 2, 31))
                    first_frame = np.full((20, 20, 3), 3, dtype=np.uint8)
                    self.assertTrue(detector.submit(first_frame, 3, 31))
                    self.assertTrue(first_started.wait(timeout=2))

                    for frame_index in (6, 9, 12):
                        frame = np.full((20, 20, 3), frame_index, dtype=np.uint8)
                        self.assertTrue(detector.submit(frame, frame_index, 31))
                    release_first.set()
                    self.assertTrue(second_started.wait(timeout=2))
                    detector.close()

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(int(model.calls[0]["source"][0, 0, 0]), 3)
        self.assertEqual(int(model.calls[1]["source"][0, 0, 0]), 12)
        self.assertEqual(model.calls[0]["conf"], 0.45)
        self.assertEqual(model.calls[0]["classes"], [0, 1, 2, 3, 5, 7])
        self.assertEqual(model.calls[0]["device"], "mps")
        self.assertFalse(detector._thread.is_alive())

    def test_submission_cadence_tracks_last_submission_when_stride_changes(self):
        class IdleModel:
            def predict(self, **_kwargs):
                return []

        with patch("lanetalk.detector._load_yolo_model", return_value=IdleModel()):
            with patch("lanetalk.detector._select_device", return_value="cpu"):
                with patch("lanetalk.detector._warmup_model"):
                    detector = ObjectDetector()
                    detector.start()
                    self.assertTrue(detector.wait_until_ready(2))
                    detector._current_stride = 5
                    frame = np.zeros((20, 20, 3), dtype=np.uint8)

                    self.assertTrue(detector.submit(frame, 5, 31))
                    self.assertFalse(detector.submit(frame, 8, 31))
                    self.assertTrue(detector.submit(frame, 10, 31))
                    detector.close()

    def test_mps_inference_failure_retries_on_cpu(self):
        class MpsLimitedModel:
            def __init__(self):
                self.devices = []

            def predict(self, **kwargs):
                self.devices.append(kwargs["device"])
                if kwargs["device"] == "mps":
                    raise RuntimeError("MPS operation unavailable")
                return []

        model = MpsLimitedModel()
        with patch("lanetalk.detector._load_yolo_model", return_value=model):
            with patch("lanetalk.detector._select_device", return_value="mps"):
                with patch("lanetalk.detector._warmup_model"):
                    detector = ObjectDetector()
                    detector.start()
                    self.assertTrue(detector.wait_until_ready(2))
                    detector.submit(np.zeros((20, 20, 3), dtype=np.uint8), 3, 31)
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        snapshot = detector.snapshot()
                        if snapshot.average_inference_ms > 0:
                            break
                        time.sleep(0.01)
                    snapshot = detector.snapshot()
                    detector.close()

        self.assertEqual(snapshot.status, "ready")
        self.assertEqual(snapshot.device, "cpu")
        self.assertEqual(model.devices, ["mps", "cpu"])


class LaneDetectorTests(unittest.TestCase):
    def setUp(self):
        self.centered = _road_lane_frame(((100, 342), (280, 215)), ((540, 342), (360, 215)))
        self.left_offset = _road_lane_frame(((32, 342), (280, 215)), ((470, 342), (360, 215)))
        self.right_offset = _road_lane_frame(((200, 342), (300, 215)), ((556, 342), (360, 215)))

    def test_converging_gray_interior_edges_are_rejected(self):
        interior = np.full((360, 640, 3), 100, dtype=np.uint8)
        cv2.line(interior, (100, 342), (280, 215), (145, 145, 145), 7)
        cv2.line(interior, (540, 342), (360, 215), (145, 145, 145), 7)
        detector = LaneDetector()

        results = [detector.process(interior) for _ in range(5)]

        self.assertTrue(all(not result.valid for result in results))
        self.assertTrue(all(result.lane_center is None for result in results))
        self.assertTrue(all(result.rejection_reason == "low_marking_evidence" for result in results))

    def test_bright_converging_edges_on_blank_frame_fail_road_context(self):
        blank_interior = _lane_frame(((100, 342), (280, 215)), ((540, 342), (360, 215)))
        detector = LaneDetector()

        results = [detector.process(blank_interior) for _ in range(5)]

        self.assertTrue(all(not result.valid for result in results))
        self.assertTrue(all(result.rejection_reason == "low_road_context" for result in results))
        self.assertTrue(all(result.lane_center is None for result in results))

    def test_horizontal_dashboard_and_vertical_edges_are_rejected(self):
        horizontal = np.full((360, 640, 3), 70, dtype=np.uint8)
        cv2.line(horizontal, (80, 260), (560, 260), (240, 240, 240), 8)
        cv2.line(horizontal, (60, 310), (580, 310), (240, 240, 240), 8)
        vertical = np.full((360, 640, 3), 70, dtype=np.uint8)
        cv2.line(vertical, (150, 190), (150, 328), (240, 240, 240), 8)
        cv2.line(vertical, (490, 190), (490, 328), (240, 240, 240), 8)

        for frame in (horizontal, vertical):
            result = LaneDetector().process(frame)
            self.assertFalse(result.valid)
            self.assertIsNone(result.lane_center)
            self.assertIsNone(result.offset)

    def test_hood_edges_that_diverge_upward_are_rejected(self):
        hood = np.full((360, 640, 3), 85, dtype=np.uint8)
        cv2.line(hood, (220, 328), (80, 194), (240, 240, 240), 8)
        cv2.line(hood, (420, 328), (560, 194), (240, 240, 240), 8)

        result = LaneDetector().process(hood)

        self.assertFalse(result.valid)
        self.assertIsNone(result.lane_center)
        self.assertIn(result.rejection_reason, {"no_left_lane", "no_right_lane", "vanishing_point"})

    def test_swapped_left_right_edge_directions_are_rejected(self):
        swapped = np.full((360, 640, 3), 85, dtype=np.uint8)
        cv2.line(swapped, (520, 328), (420, 194), (245, 245, 245), 8)
        cv2.line(swapped, (120, 328), (220, 194), (245, 245, 245), 8)

        result = LaneDetector().process(swapped)

        self.assertFalse(result.valid)
        self.assertIsNone(result.lane_center)
        self.assertIn(result.rejection_reason, {"no_left_lane", "no_right_lane", "vanishing_point"})

    def test_weak_hough_support_is_rejected(self):
        weak_segments = np.array([[[170, 328, 310, 194]], [[470, 328, 330, 194]]], dtype=np.int32)
        with patch("lanetalk.detector.cv2.HoughLinesP", return_value=weak_segments):
            result = LaneDetector().process(self.centered)

        self.assertFalse(result.valid)
        self.assertEqual(result.rejection_reason, "weak_support")

    def test_white_markings_become_trusted_after_three_consistent_frames(self):
        detector = LaneDetector()

        results = [detector.process(self.centered) for _ in range(3)]

        self.assertEqual([result.valid for result in results], [False, False, True])
        self.assertEqual([result.trusted_count for result in results], [1, 2, 3])
        self.assertIsNone(results[0].lane_center)
        self.assertIsNone(results[1].offset)
        self.assertAlmostEqual(results[2].lane_center, 320, delta=10)
        self.assertAlmostEqual(results[2].offset, 0, delta=0.05)

    def test_yellow_markings_can_support_a_valid_lane(self):
        yellow = _road_lane_frame(
            ((100, 342), (280, 215)),
            ((540, 342), (360, 215)),
            color=(0, 220, 255),
        )
        detector = LaneDetector()

        results = [detector.process(yellow) for _ in range(3)]

        self.assertTrue(results[-1].valid)
        self.assertGreater(results[-1].marking_evidence, 0)

    def test_color_is_supporting_evidence_not_a_hard_requirement(self):
        textured = np.full((360, 640, 3), 82, dtype=np.uint8)
        for y in range(195, 340, 12):
            level = 50 if y % 24 else 125
            cv2.line(textured, (0, y), (639, y), (level, level, level), 3)
        cv2.line(textured, (100, 342), (280, 215), (145, 145, 145), 7)
        cv2.line(textured, (540, 342), (360, 215), (145, 145, 145), 7)
        detector = LaneDetector()

        results = [detector.process(textured) for _ in range(3)]

        self.assertTrue(results[-1].valid)
        self.assertEqual(results[-1].marking_evidence, 0.0)

    def test_temporal_jump_clears_trust_and_drift_persistence(self):
        detector = LaneDetector()
        for _ in range(3):
            detector.process(self.left_offset)
        trusted = detector.process(self.left_offset)
        self.assertTrue(trusted.valid)
        self.assertGreater(trusted.persistence_count, 0)

        jump = detector.process(self.right_offset)

        self.assertFalse(jump.valid)
        self.assertEqual(jump.rejection_reason, "temporal_jump")
        self.assertIsNone(jump.lane_center)
        self.assertIsNone(jump.offset)
        self.assertEqual(jump.persistence_count, 0)

    def test_centered_synthetic_lane_has_near_zero_offset_without_drift(self):
        detector = LaneDetector()
        result = None
        for _ in range(3):
            result = detector.process(self.centered)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.left_line)
        self.assertIsNotNone(result.right_line)
        self.assertAlmostEqual(result.lane_center, 320, delta=10)
        self.assertAlmostEqual(result.offset, 0, delta=0.05)
        self.assertAlmostEqual(
            result.offset,
            (result.lane_center - result.camera_center) / result.camera_center,
            delta=0.01,
        )
        self.assertGreater(result.confidence, 0)
        self.assertEqual(result.drift_state, "center")
        self.assertEqual(result.events, [])

    def test_pipeline_exposes_roi_hough_and_accepted_candidates(self):
        frame = self.centered.copy()
        cv2.line(frame, (0, 30), (100, 80), (255, 255, 255), 5)
        with patch("lanetalk.detector.cv2.cvtColor", wraps=cv2.cvtColor) as grayscale:
            with patch("lanetalk.detector.cv2.GaussianBlur", wraps=cv2.GaussianBlur) as blur:
                with patch("lanetalk.detector.cv2.Canny", wraps=cv2.Canny) as canny:
                    with patch("lanetalk.detector.cv2.HoughLinesP", wraps=cv2.HoughLinesP) as hough:
                        result = LaneDetector().process(frame)

        self.assertEqual(grayscale.call_count, 2)  # grayscale edges and HLS marking mask
        self.assertEqual(grayscale.call_args_list[0].args[1], cv2.COLOR_BGR2GRAY)
        self.assertEqual(grayscale.call_args_list[1].args[1], cv2.COLOR_BGR2HLS)
        blur.assert_called_once()
        self.assertEqual(blur.call_args.args[1], (5, 5))
        canny.assert_called_once()
        hough.assert_called_once()
        self.assertEqual(result.roi_polygon, ((51, 328), (256, 194), (384, 194), (589, 328)))
        self.assertGreater(len(result.raw_segments), 0)
        self.assertGreater(len(result.left_candidates), 0)
        self.assertGreater(len(result.right_candidates), 0)
        hough_input = hough.call_args.args[0]
        self.assertEqual(cv2.countNonZero(hough_input[:194]), 0)

        debug_frame = self.centered.copy()
        LaneDetector.draw(debug_frame, result, debug=True)
        self.assertTupleEqual(tuple(debug_frame[328, 51]), (0, 255, 255))

    def test_blank_frame_has_no_lane_center_or_event(self):
        result = LaneDetector().process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertFalse(result.valid)
        self.assertIsNone(result.lane_center)
        self.assertIsNone(result.offset)
        self.assertEqual(result.persistence_count, 0)
        self.assertEqual(result.events, [])

    def test_single_lane_line_never_invents_a_center_or_drift_event(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.line(frame, (100, 342), (280, 215), (255, 255, 255), 6)
        detector = LaneDetector()

        for _ in range(10):
            result = detector.process(frame)
            self.assertIsNone(result.lane_center)
            self.assertIsNone(result.offset)
            self.assertFalse(result.valid)
            self.assertGreater(len(result.left_candidates), 0)
            self.assertEqual(len(result.right_candidates), 0)
            self.assertEqual(result.confidence, 0.0)
            self.assertEqual(result.events, [])

    def test_left_shift_has_negative_offset_and_left_state(self):
        detector = LaneDetector()
        result = None
        for _ in range(3):
            result = detector.process(self.left_offset)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.left_line)
        self.assertIsNotNone(result.right_line)
        self.assertIsNotNone(result.lane_center)
        self.assertLess(result.offset, -0.15)
        self.assertEqual(result.camera_center, 320)
        self.assertEqual(result.drift_state, "left")

    def test_temporary_left_deviation_does_not_emit_before_persistence(self):
        detector = LaneDetector()
        for _ in range(7):
            result = detector.process(self.left_offset)
            self.assertEqual(result.events, [])
        self.assertEqual(result.persistence_count, 5)

    def test_eight_trusted_deviation_frames_are_needed_for_drift_event(self):
        detector = LaneDetector()
        results = [detector.process(self.left_offset) for _ in range(9)]

        self.assertTrue(all(not result.events for result in results))
        self.assertEqual(results[-1].persistence_count, 7)
        result = detector.process(self.left_offset)
        self.assertEqual([event.details["direction"] for event in result.events], ["left"])

    def test_sustained_left_drift_emits_and_respects_three_second_cooldown(self):
        detector = LaneDetector()
        monotonic_values = iter([10.0] * 8 + [12.9] * 4 + [13.0])
        with patch("lanetalk.detector.time.monotonic", side_effect=lambda: next(monotonic_values)):
            events = []
            for _ in range(10):
                events.extend(detector.process(self.left_offset).events)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].kind, "lane_drift")
            self.assertEqual(events[0].details["direction"], "left")
            self.assertLess(events[0].details["offset"], -0.15)

            for _ in range(4):
                self.assertEqual(detector.process(self.left_offset).events, [])
            self.assertEqual(detector.process(self.left_offset).events[0].kind, "lane_drift")

    def test_sustained_right_drift_emits_right_event(self):
        detector = LaneDetector()
        monotonic_values = iter([20.0] * 20)
        with patch("lanetalk.detector.time.monotonic", side_effect=lambda: next(monotonic_values)):
            results = [detector.process(self.right_offset) for _ in range(10)]

        self.assertEqual([result.valid for result in results[:2]], [False, False])
        self.assertTrue(all(result.valid for result in results[2:]))
        self.assertGreater(results[-1].offset, 0.15)
        self.assertEqual(results[-1].drift_state, "right")
        self.assertEqual([event.details["direction"] for result in results for event in result.events], ["right"])

    def test_invalid_lane_frame_resets_persistence(self):
        detector = LaneDetector()
        for _ in range(7):
            self.assertEqual(detector.process(self.left_offset).events, [])

        invalid = detector.process(np.zeros_like(self.left_offset))
        self.assertFalse(invalid.valid)
        self.assertIsNone(invalid.lane_center)
        self.assertIsNone(invalid.offset)
        self.assertEqual(invalid.persistence_count, 0)
        for _ in range(9):
            self.assertEqual(detector.process(self.left_offset).events, [])
        self.assertEqual(len(detector.process(self.left_offset).events), 1)

    def test_camera_gap_resets_lane_persistence(self):
        detector = LaneDetector()
        for _ in range(9):
            self.assertEqual(detector.process(self.left_offset).events, [])

        detector.reset()
        for _ in range(9):
            self.assertEqual(detector.process(self.left_offset).events, [])
        self.assertEqual(len(detector.process(self.left_offset).events), 1)

    def test_implausibly_wide_lane_pair_is_rejected(self):
        wide_pair = _lane_frame(((5, 342), (280, 215)), ((635, 342), (360, 215)))
        result = LaneDetector().process(wide_pair)

        self.assertIsNone(result.lane_center)
        self.assertIsNone(result.offset)
        self.assertEqual(result.events, [])

    def test_smoothing_moves_offset_toward_center(self):
        detector = LaneDetector()
        before = None
        for _ in range(3):
            before = detector.process(self.left_offset).offset
        after = None
        for _ in range(3):
            after = detector.process(self.centered).offset

        self.assertLess(abs(after), abs(before))


class AppTests(unittest.TestCase):
    def test_lane_debug_cli_flag_is_available(self):
        from lanetalk.app import _parse_args

        self.assertTrue(_parse_args(["--lane-debug"]).lane_debug)

    def test_lane_roi_cli_accepts_normalized_four_point_polygon(self):
        from lanetalk.app import _parse_args

        args = _parse_args(["--lane-debug", "--lane-roi", "0.08,0.91,0.40,0.54,0.60,0.54,0.92,0.91"])

        self.assertEqual(args.lane_roi, ((0.08, 0.91), (0.40, 0.54), (0.60, 0.54), (0.92, 0.91)))

    def test_q_closes_camera_worker_and_window(self):
        from lanetalk import app
        from lanetalk.detector import LaneResult, ObjectDetectorSnapshot

        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        class FakeCamera:
            last_error = ""
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

            def read(self):
                return frame.copy()

            def close(self):
                self.closed = True

        class FakeObjects:
            closed = False

            def start(self):
                pass

            def submit(self, *_args):
                return False

            def snapshot(self):
                return ObjectDetectorSnapshot(status="ready", device="cpu")

            def drain_events(self):
                return []

            def close(self):
                self.closed = True

        class FakeLanes:
            def process(self, _frame):
                return LaneResult(None, None, None, 320.0, None, 0.2)

            @staticmethod
            def draw(_frame, _result):
                pass

        camera = FakeCamera()
        objects = FakeObjects()
        with patch.object(app, "CameraStream", return_value=camera):
            with patch.object(app, "ObjectDetector", return_value=objects):
                with patch.object(app, "LaneDetector", return_value=FakeLanes()):
                    with patch.object(app.cv2, "imshow"):
                        with patch.object(app.cv2, "waitKey", return_value=ord("q")):
                            with patch.object(app.cv2, "destroyAllWindows") as destroy:
                                app.main([])

        self.assertTrue(camera.closed)
        self.assertTrue(objects.closed)
        destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
