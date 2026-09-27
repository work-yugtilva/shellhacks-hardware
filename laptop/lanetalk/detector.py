"""Local object and lane detectors for the LaneTalk camera pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lanetalk.events import Event


_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _load_yolo_model(model_path: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(model_path))


def _select_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.backends.mps.is_available():
            return "mps"
    except (AttributeError, RuntimeError):
        pass
    return "cpu"


def _warmup_model(model: Any, device: str, confidence: float) -> None:
    warmup_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    model.predict(
        source=warmup_frame,
        conf=confidence,
        classes=list(_COCO_CLASSES),
        device=device,
        verbose=False,
    )


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ObjectDetectorSnapshot:
    detections: tuple[Detection, ...] = ()
    last_inference_ms: float = 0.0
    average_inference_ms: float = 0.0
    stride: int = 3
    device: str = "loading"
    status: str = "loading"
    error: str = ""


class ObjectDetector:
    """Run YOLO11n on a background worker without building a frame backlog."""

    def __init__(
        self,
        model_path: str | Path = Path(__file__).resolve().parents[1] / "yolo11n.pt",
        confidence: float = 0.45,
        base_stride: int = 3,
        cooldown_seconds: float = 2.0,
        iou_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if base_stride < 1:
            raise ValueError("base_stride must be at least 1")

        self.model_path = model_path
        self.confidence = confidence
        self.base_stride = base_stride
        self.cooldown_seconds = cooldown_seconds
        self.iou_threshold = iou_threshold

        self._condition = threading.Condition()
        self._result_lock = threading.Lock()
        self._pending: tuple[np.ndarray, float] | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._events: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._snapshot = ObjectDetectorSnapshot(stride=base_stride)
        self._current_stride = base_stride
        self._average_frame_period_ms = 1000.0 / 30.0
        self._average_inference_ms = 0.0
        self._recent_boxes: list[tuple[str, tuple[int, int, int, int], float]] = []
        self._last_submitted_frame = 0

    def start(self) -> None:
        """Start loading the model and processing submitted frames."""
        with self._condition:
            if self._thread is not None:
                return
            if self._stop:
                raise RuntimeError("cannot start a closed ObjectDetector")
            self._thread = threading.Thread(
                target=self._worker,
                name="lanetalk-yolo",
                daemon=True,
            )
            self._thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait for model loading to finish; primarily useful for diagnostics."""
        return self._ready.wait(timeout)

    def submit(self, frame: np.ndarray, frame_index: int, frame_period_ms: float) -> bool:
        """Offer every stride-th frame; replace any pending frame with the newest."""
        with self._condition:
            if self._thread is None or self._stop:
                return False
            if self._snapshot.status == "error":
                return False
            if frame_period_ms > 0:
                self._average_frame_period_ms = (
                    0.8 * self._average_frame_period_ms + 0.2 * frame_period_ms
                )
            if frame_index - self._last_submitted_frame < self._current_stride:
                return False

            self._pending = (frame.copy(), self._average_frame_period_ms)
            self._last_submitted_frame = frame_index
            self._condition.notify()
            return True

    def snapshot(self) -> ObjectDetectorSnapshot:
        with self._result_lock:
            return self._snapshot

    def drain_events(self) -> list[Event]:
        events: list[Event] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()

    @staticmethod
    def _stride_for_timing(base_stride: int, inference_ms: float, frame_period_ms: float) -> int:
        if frame_period_ms <= 0:
            return base_stride
        return max(base_stride, math.ceil(inference_ms / frame_period_ms))

    @staticmethod
    def _detections_from_results(results: Any) -> list[Detection]:
        detections: list[Detection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = _to_numpy(boxes.xyxy).reshape(-1, 4)
            confidences = _to_numpy(boxes.conf).reshape(-1)
            class_ids = _to_numpy(boxes.cls).reshape(-1)
            for bbox, confidence, class_id in zip(coordinates, confidences, class_ids):
                class_name = _COCO_CLASSES.get(int(class_id))
                if class_name is None:
                    continue
                pixel_bbox = tuple(int(round(value)) for value in bbox)
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=pixel_bbox,  # type: ignore[arg-type]
                    )
                )
        return detections

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _events_for_detections(
        self,
        detections: list[Detection],
        now_monotonic: float,
        timestamp: float,
    ) -> list[Event]:
        self._recent_boxes = [
            item for item in self._recent_boxes
            if now_monotonic - item[2] <= self.cooldown_seconds
        ]
        emitted: list[Event] = []
        for detection in detections:
            match = next(
                (
                    index
                    for index, (class_name, bbox, _) in enumerate(self._recent_boxes)
                    if class_name == detection.class_name
                    and self._iou(bbox, detection.bbox) >= self.iou_threshold
                ),
                None,
            )
            if match is not None:
                self._recent_boxes[match] = (
                    detection.class_name,
                    detection.bbox,
                    now_monotonic,
                )
                continue

            self._recent_boxes.append((detection.class_name, detection.bbox, now_monotonic))
            kind = "pedestrian_detected" if detection.class_name == "person" else "vehicle_detected"
            emitted.append(
                Event(
                    kind=kind,
                    confidence=detection.confidence,
                    timestamp=timestamp,
                    details={"class": detection.class_name, "bbox": list(detection.bbox)},
                )
            )
        return emitted

    def _set_status(self, status: str, device: str = "loading", error: str = "") -> None:
        with self._result_lock:
            previous = self._snapshot
            self._snapshot = ObjectDetectorSnapshot(
                detections=() if status == "error" else previous.detections,
                last_inference_ms=previous.last_inference_ms,
                average_inference_ms=previous.average_inference_ms,
                stride=previous.stride,
                device=device,
                status=status,
                error=error,
            )
        if status in {"ready", "error"}:
            self._ready.set()

    def _worker(self) -> None:
        device = "cpu"
        try:
            model = _load_yolo_model(self.model_path)
            device = _select_device()
            self._set_status("warming up", device=device)
            try:
                _warmup_model(model, device, self.confidence)
            except Exception:
                if device != "mps":
                    raise
                device = "cpu"
                _warmup_model(model, device, self.confidence)
        except Exception as exc:
            self._set_status("error", device=device, error=f"{type(exc).__name__}: {exc}")
            return

        self._set_status("ready", device=device)
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                frame, frame_period_ms = self._pending
                self._pending = None

            started = time.perf_counter()
            try:
                results = model.predict(
                    source=frame,
                    conf=self.confidence,
                    classes=list(_COCO_CLASSES),
                    device=device,
                    verbose=False,
                )
                detections = self._detections_from_results(results)
            except Exception as exc:
                if device != "mps":
                    self._set_status("error", device=device, error=f"{type(exc).__name__}: {exc}")
                    return
                device = "cpu"
                try:
                    results = model.predict(
                        source=frame,
                        conf=self.confidence,
                        classes=list(_COCO_CLASSES),
                        device=device,
                        verbose=False,
                    )
                    detections = self._detections_from_results(results)
                except Exception as cpu_exc:
                    self._set_status("error", device=device, error=f"{type(cpu_exc).__name__}: {cpu_exc}")
                    return

            inference_ms = (time.perf_counter() - started) * 1000
            self._average_inference_ms = (
                inference_ms
                if self._average_inference_ms == 0
                else 0.5 * self._average_inference_ms + 0.5 * inference_ms
            )
            average_ms = self._average_inference_ms
            now_monotonic = time.monotonic()
            timestamp = time.time()
            events = self._events_for_detections(detections, now_monotonic, timestamp)

            with self._condition:
                self._current_stride = self._stride_for_timing(
                    self.base_stride,
                    average_ms,
                    frame_period_ms,
                )
                stride = self._current_stride

            with self._result_lock:
                self._snapshot = ObjectDetectorSnapshot(
                    detections=tuple(detections),
                    last_inference_ms=inference_ms,
                    average_inference_ms=average_ms,
                    stride=stride,
                    device=device,
                    status="ready",
                )
            for event in events:
                self._events.put(event)


@dataclass(frozen=True, slots=True)
class LaneValidityConfig:
    """Conservative lane gates; ROI vertices are normalized x/y points.

    The default trapezoid is a generic starter, not a camera calibration.
    Width, slope, support, fit, appearance, and temporal limits are tuneable.
    """

    roi_vertices: tuple[tuple[float, float], ...] = (
        (0.08, 0.91), (0.40, 0.54), (0.60, 0.54), (0.92, 0.91)
    )
    min_abs_slope: float = 0.35
    max_abs_slope: float = 4.0
    lane_width_range: tuple[float, float] = (0.25, 0.78)
    vanishing_x_range: tuple[float, float] = (0.25, 0.75)
    vanishing_y_range: tuple[float, float] = (0.18, 0.72)
    convergence_ratio_range: tuple[float, float] = (0.01, 0.80)
    min_candidate_segments: int = 2
    min_support_height_fraction: float = 0.18
    max_fit_residual_width_fraction: float = 0.035
    max_temporal_jump_width_fraction: float = 0.14
    white_light_min: int = 170
    white_saturation_max: int = 110
    yellow_hue_range: tuple[int, int] = (12, 45)
    yellow_saturation_min: int = 65
    yellow_light_min: int = 70
    min_surface_texture_std: float = 0.8
    fused_confidence_floor: float = 0.66
    trust_frames: int = 3

    def __post_init__(self) -> None:
        if len(self.roi_vertices) != 4 or any(
            len(point) != 2 or not all(0 <= value <= 1 for value in point)
            for point in self.roi_vertices
        ):
            raise ValueError("roi_vertices must contain four normalized x/y points")
        if not 0 < self.min_abs_slope < self.max_abs_slope:
            raise ValueError("lane slope bounds are invalid")
        if self.trust_frames < 1:
            raise ValueError("trust_frames must be at least 1")


@dataclass(slots=True)
class LaneResult:
    left_line: tuple[int, int, int, int] | None
    right_line: tuple[int, int, int, int] | None
    lane_center: float | None
    camera_center: float
    offset: float | None
    processing_ms: float
    events: list[Event] = field(default_factory=list)
    roi_polygon: tuple[tuple[int, int], ...] = ()
    raw_segments: tuple[tuple[int, int, int, int], ...] = ()
    left_candidates: tuple[tuple[int, int, int, int], ...] = ()
    right_candidates: tuple[tuple[int, int, int, int], ...] = ()
    valid: bool = False
    confidence: float = 0.0
    persistence_count: int = 0
    persistence_required: int = 8
    drift_state: str = "invalid"
    geometry_valid: bool = False
    candidate_center: float | None = None
    candidate_offset: float | None = None
    trusted_count: int = 0
    trust_required: int = 3
    left_support: float = 0.0
    right_support: float = 0.0
    lane_width_normalized: float | None = None
    marking_evidence: float = 0.0
    surface_texture_std: float = 0.0
    rejection_reason: str | None = None
    vanishing_point: tuple[float, float] | None = None


class LaneDetector:
    """Estimate and temporally validate classical lane geometry."""

    def __init__(
        self,
        deviation_threshold: float = 0.15,
        persistence_frames: int = 8,
        cooldown_seconds: float = 3.0,
        smoothing_alpha: float = 0.25,
        config: LaneValidityConfig | None = None,
    ) -> None:
        if deviation_threshold <= 0:
            raise ValueError("deviation_threshold must be positive")
        if persistence_frames < 1:
            raise ValueError("persistence_frames must be at least 1")
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.deviation_threshold = deviation_threshold
        self.persistence_frames = persistence_frames
        self.cooldown_seconds = cooldown_seconds
        self.smoothing_alpha = smoothing_alpha
        self.config = config or LaneValidityConfig()
        self._smoothed_offset: float | None = None
        self._offset_history: deque[float] = deque(maxlen=persistence_frames)
        self._deviation_direction: str | None = None
        self._deviation_frames = 0
        self._last_event_at = float("-inf")
        self._candidate_history: deque[tuple[float, float, float, float]] = deque(
            maxlen=self.config.trust_frames
        )

    @staticmethod
    def _classify_segments(
        segments: np.ndarray | None,
        width: int,
        height: int = 360,
        y_top: int | None = None,
        config: LaneValidityConfig | None = None,
    ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        raw: list[tuple[int, int, int, int]] = []
        left: list[tuple[int, int, int, int]] = []
        right: list[tuple[int, int, int, int]] = []
        if segments is None:
            return raw, left, right
        validity = config or LaneValidityConfig()
        top_limit = int(height * 0.54) if y_top is None else y_top
        for values in segments.reshape(-1, 4):
            line = tuple(int(value) for value in values)
            raw.append(line)  # type: ignore[arg-type]
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            slope = dy / dx
            length = math.hypot(dx, dy)
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            if (
                length < max(12, height * 0.06)
                or midpoint_y < top_limit + height * 0.08
                or midpoint_x < width * 0.04
                or midpoint_x > width * 0.96
                or not validity.min_abs_slope <= abs(slope) <= validity.max_abs_slope
            ):
                continue
            if slope < 0 and midpoint_x < width / 2:
                left.append(line)  # type: ignore[arg-type]
            elif slope > 0 and midpoint_x > width / 2:
                right.append(line)  # type: ignore[arg-type]
        return raw, left, right

    @staticmethod
    def _fit_candidates(
        candidates: list[tuple[int, int, int, int]], side: str, width: int, y_top: int, y_bottom: int
    ) -> tuple[int, int, int, int] | None:
        fits: list[tuple[float, float, float]] = []
        for x1, y1, x2, y2 in candidates:
            dx = x2 - x1
            if dx == 0:
                continue
            slope = (y2 - y1) / dx
            length = math.hypot(dx, y2 - y1)
            fits.append((slope, y1 - slope * x1, length))
        if not fits:
            return None
        total = sum(weight for _, _, weight in fits)
        slope = sum(value * weight for value, _, weight in fits) / total
        intercept = sum(value * weight for _, value, weight in fits) / total
        if slope == 0:
            return None
        x_top = int(round((y_top - intercept) / slope))
        x_bottom = int(round((y_bottom - intercept) / slope))
        if side == "left" and not (0 <= x_bottom < width / 2 and 0 <= x_top < width):
            return None
        if side == "right" and not (width / 2 < x_bottom < width and 0 <= x_top < width):
            return None
        return (x_top, y_top, x_bottom, y_bottom)

    @staticmethod
    def _fit_residual_fraction(
        candidates: list[tuple[int, int, int, int]], line: tuple[int, int, int, int], width: int
    ) -> float:
        x1, y1, x2, y2 = line
        slope = (x2 - x1) / max(1, y2 - y1)
        errors = []
        for ax, ay, bx, by in candidates:
            errors.extend((abs(ax - (x1 + (ay - y1) * slope)), abs(bx - (x1 + (by - y1) * slope))))
        return float(np.mean(errors) / width) if errors else float("inf")

    @staticmethod
    def _x_at_y(line: tuple[int, int, int, int], y: float) -> float:
        x1, y1, x2, y2 = line
        return float(x1) if y1 == y2 else x1 + (y - y1) * (x2 - x1) / (y2 - y1)

    @staticmethod
    def _line_slope(line: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        return (y2 - y1) / (x2 - x1) if x2 != x1 else float("inf")

    def _reset_track(self) -> None:
        self._candidate_history.clear()
        self._smoothed_offset = None
        self._offset_history.clear()
        self._deviation_direction = None
        self._deviation_frames = 0

    def reset(self) -> None:
        """Forget lane trust after a camera gap while retaining event cooldown."""
        self._reset_track()

    def _marking_mask(self, frame: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(
            hls,
            np.array([0, self.config.white_light_min, 0], dtype=np.uint8),
            np.array([180, 255, self.config.white_saturation_max], dtype=np.uint8),
        )
        yellow = cv2.inRange(
            hls,
            np.array(
                [self.config.yellow_hue_range[0], self.config.yellow_light_min, self.config.yellow_saturation_min],
                dtype=np.uint8,
            ),
            np.array([self.config.yellow_hue_range[1], 255, 255], dtype=np.uint8),
        )
        return cv2.bitwise_or(white, yellow)

    @staticmethod
    def _corridor_mask(line: tuple[int, int, int, int], shape: tuple[int, int], roi_mask: np.ndarray) -> np.ndarray:
        height, width = shape
        corridor = np.zeros((height, width), dtype=np.uint8)
        cv2.line(corridor, line[:2], line[2:], 255, thickness=max(11, int(width * 0.035)))
        return cv2.bitwise_and(corridor, roi_mask)

    def _appearance_evidence(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        left_line: tuple[int, int, int, int],
        right_line: tuple[int, int, int, int],
        roi_mask: np.ndarray,
    ) -> tuple[float, float]:
        height, width = gray.shape
        marking_mask = self._marking_mask(frame)
        ratios = []
        for line in (left_line, right_line):
            corridor = self._corridor_mask(line, (height, width), roi_mask)
            area = cv2.countNonZero(corridor)
            paint = cv2.countNonZero(cv2.bitwise_and(marking_mask, corridor))
            ratios.append(paint / area if area else 0.0)
        marking_evidence = float(np.mean(ratios))

        interior = np.zeros((height, width), dtype=np.uint8)
        y_top, y_bottom = int(height * 0.66), int(height * 0.90)
        polygon = np.array([[
            (int(self._x_at_y(left_line, y_top)), y_top),
            (int(self._x_at_y(right_line, y_top)), y_top),
            (int(self._x_at_y(right_line, y_bottom)), y_bottom),
            (int(self._x_at_y(left_line, y_bottom)), y_bottom),
        ]], dtype=np.int32)
        cv2.fillPoly(interior, polygon, 255)
        interior = cv2.bitwise_and(interior, roi_mask)
        for line in (left_line, right_line):
            interior = cv2.bitwise_and(
                interior,
                cv2.bitwise_not(self._corridor_mask(line, (height, width), roi_mask)),
            )
        pixels = gray[interior > 0]
        return marking_evidence, float(np.std(pixels)) if pixels.size else 0.0

    def _confidence(
        self,
        left_support: float,
        right_support: float,
        left_count: int,
        right_count: int,
        width: int,
        height: int,
        lane_width: float,
        marking: float,
        texture_std: float,
    ) -> float:
        low, high = self.config.lane_width_range
        width_score = max(0.0, 1 - abs(lane_width - (low + high) / 2) / ((high - low) / 2))
        geometry_score = 0.7 + 0.3 * width_score
        weaker = min(left_support, right_support)
        support_score = min(1.0, weaker / max(1.0, height * 0.45))
        count_score = min(1.0, min(left_count, right_count) / 4.0)
        balance = weaker / max(left_support, right_support, 1.0)
        edge_score = 0.6 * support_score + 0.25 * count_score + 0.15 * balance
        marking_score = min(1.0, marking / 0.24)
        texture_score = min(1.0, texture_std / 12.0)
        return min(1.0, 0.25 * geometry_score + 0.30 * edge_score + 0.25 * marking_score + 0.20 * texture_score)

    def process(self, frame: np.ndarray) -> LaneResult:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        camera_center = width / 2
        roi_polygon = tuple((int(round(x * width)), int(round(y * height))) for x, y in self.config.roi_vertices)
        roi = np.array([roi_polygon], dtype=np.int32)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, roi, 255)
        y_top, y_bottom = min(y for _, y in roi_polygon), max(y for _, y in roi_polygon)
        anchor_y = min(int(height * 0.90), y_bottom)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        min_line_length = max(12, int(height * 0.04))
        segments = cv2.HoughLinesP(
            masked_edges, 1, np.pi / 180, threshold=30,
            minLineLength=min_line_length, maxLineGap=max(8, min_line_length),
        )
        raw, left_candidates, right_candidates = self._classify_segments(
            segments, width, height, y_top, self.config
        )
        left_line = self._fit_candidates(left_candidates, "left", width, y_top, y_bottom)
        right_line = self._fit_candidates(right_candidates, "right", width, y_top, y_bottom)
        left_support = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in left_candidates)
        right_support = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in right_candidates)
        geometry_valid = False
        reason: str | None = None
        confidence = marking_evidence = texture_std = 0.0
        width_normalized: float | None = None
        vanishing_point: tuple[float, float] | None = None
        left_x = right_x = 0.0

        if left_line is None or not left_candidates:
            reason = "no_left_lane"
        elif right_line is None or not right_candidates:
            reason = "no_right_lane"
        elif min(len(left_candidates), len(right_candidates)) < self.config.min_candidate_segments:
            reason = "weak_support"
        elif min(left_support, right_support) < height * self.config.min_support_height_fraction:
            reason = "weak_support"
        elif max(
            self._fit_residual_fraction(left_candidates, left_line, width),
            self._fit_residual_fraction(right_candidates, right_line, width),
        ) > self.config.max_fit_residual_width_fraction:
            reason = "fit_residual"
        else:
            ls, rs = self._line_slope(left_line), self._line_slope(right_line)
            if not (ls < 0 < rs and self.config.min_abs_slope <= abs(ls) <= self.config.max_abs_slope
                    and self.config.min_abs_slope <= abs(rs) <= self.config.max_abs_slope):
                reason = "bad_slope"
            else:
                left_x, right_x = self._x_at_y(left_line, anchor_y), self._x_at_y(right_line, anchor_y)
                lane_width = right_x - left_x
                width_normalized = lane_width / width
                top_width = self._x_at_y(right_line, y_top) - self._x_at_y(left_line, y_top)
                ratio = top_width / lane_width if lane_width > 0 else float("inf")
                if not (
                    0.05 * width < left_x < camera_center - 0.03 * width
                    and camera_center + 0.03 * width < right_x < 0.95 * width
                    and self.config.lane_width_range[0] <= width_normalized <= self.config.lane_width_range[1]
                ):
                    reason = "bad_lane_width"
                elif not (top_width > 0 and self.config.convergence_ratio_range[0] <= ratio <= self.config.convergence_ratio_range[1]):
                    reason = "vanishing_point"
                else:
                    la = (left_line[2] - left_line[0]) / max(1, left_line[3] - left_line[1])
                    ra = (right_line[2] - right_line[0]) / max(1, right_line[3] - right_line[1])
                    if abs(la - ra) < 0.03:
                        reason = "vanishing_point"
                    else:
                        left_intercept = left_line[0] - la * left_line[1]
                        right_intercept = right_line[0] - ra * right_line[1]
                        vy = (right_intercept - left_intercept) / (la - ra)
                        vx = left_line[0] + la * (vy - left_line[1])
                        vanishing_point = (float(vx), float(vy))
                        if not (
                            self.config.vanishing_x_range[0] * width <= vx <= self.config.vanishing_x_range[1] * width
                            and self.config.vanishing_y_range[0] * height <= vy <= self.config.vanishing_y_range[1] * height
                        ):
                            reason = "vanishing_point"
                        else:
                            marking_evidence, texture_std = self._appearance_evidence(
                                frame, gray, left_line, right_line, roi_mask
                            )
                            confidence = self._confidence(
                                left_support, right_support, len(left_candidates), len(right_candidates),
                                width, height, width_normalized, marking_evidence, texture_std,
                            )
                            if texture_std < self.config.min_surface_texture_std and marking_evidence > 0.20:
                                reason = "low_road_context"
                            elif confidence < self.config.fused_confidence_floor:
                                reason = "low_marking_evidence"
                            else:
                                geometry_valid = True

        candidate_center: float | None = None
        candidate_offset: float | None = None
        center: float | None = None
        offset: float | None = None
        events: list[Event] = []
        valid = False
        drift_state = "invalid"
        trusted_count = 0
        if geometry_valid and left_line is not None and right_line is not None:
            candidate_center = (left_x + right_x) / 2
            candidate_offset = (candidate_center - camera_center) / camera_center
            key = (left_x / width, right_x / width, candidate_center / width, (width_normalized or 0.0))
            if self._candidate_history and any(
                abs(value - previous) > self.config.max_temporal_jump_width_fraction
                for value, previous in zip(key, self._candidate_history[-1])
            ):
                self._reset_track()
                reason = "temporal_jump"
            else:
                self._candidate_history.append(key)
                trusted_count = len(self._candidate_history)
                if trusted_count < self.config.trust_frames:
                    reason = "temporal_warmup"
                else:
                    valid = True
                    center = candidate_center
                    self._smoothed_offset = candidate_offset if self._smoothed_offset is None else (
                        (1 - self.smoothing_alpha) * self._smoothed_offset
                        + self.smoothing_alpha * candidate_offset
                    )
                    offset = self._smoothed_offset
                    self._offset_history.append(offset)
                    if abs(offset) >= self.deviation_threshold:
                        direction = "left" if offset < 0 else "right"
                        drift_state = direction
                        if direction == self._deviation_direction:
                            self._deviation_frames += 1
                        else:
                            self._deviation_direction = direction
                            self._deviation_frames = 1
                        now = time.monotonic()
                        if self._deviation_frames >= self.persistence_frames and now - self._last_event_at >= self.cooldown_seconds:
                            stability = max(0.0, min(1.0, 1 - float(np.std(self._offset_history)) / self.deviation_threshold))
                            events.append(Event(
                                kind="lane_drift", confidence=min(confidence, stability),
                                details={"direction": direction, "offset": float(offset)},
                            ))
                            self._last_event_at = now
                    else:
                        self._deviation_direction = None
                        self._deviation_frames = 0
                        drift_state = "center"
        else:
            self._reset_track()

        return LaneResult(
            left_line=left_line, right_line=right_line, lane_center=center,
            camera_center=camera_center, offset=offset,
            processing_ms=(time.perf_counter() - started) * 1000, events=events,
            roi_polygon=roi_polygon, raw_segments=tuple(raw),
            left_candidates=tuple(left_candidates), right_candidates=tuple(right_candidates),
            valid=valid, confidence=confidence, persistence_count=self._deviation_frames,
            persistence_required=self.persistence_frames, drift_state=drift_state,
            geometry_valid=geometry_valid, candidate_center=candidate_center,
            candidate_offset=candidate_offset, trusted_count=trusted_count,
            trust_required=self.config.trust_frames, left_support=left_support,
            right_support=right_support, lane_width_normalized=width_normalized,
            marking_evidence=marking_evidence, surface_texture_std=texture_std,
            rejection_reason=reason, vanishing_point=vanishing_point,
        )

    @staticmethod
    def draw(frame: np.ndarray, result: LaneResult, debug: bool = False) -> None:
        height, width = frame.shape[:2]
        if debug:
            if result.roi_polygon:
                cv2.polylines(frame, np.array([result.roi_polygon], dtype=np.int32), True, (0, 255, 255), 2)
            for segment in result.raw_segments:
                cv2.line(frame, segment[:2], segment[2:], (105, 105, 105), 1)
            for segment in result.left_candidates:
                cv2.line(frame, segment[:2], segment[2:], (255, 255, 0), 2)
            for segment in result.right_candidates:
                cv2.line(frame, segment[:2], segment[2:], (255, 0, 255), 2)

        left_color, right_color = ((0, 220, 0), (0, 140, 255)) if result.valid else ((0, 180, 255), (255, 180, 0))
        if result.left_line is not None:
            cv2.line(frame, result.left_line[:2], result.left_line[2:], left_color, 3)
        if result.right_line is not None:
            cv2.line(frame, result.right_line[:2], result.right_line[2:], right_color, 3)
        if result.vanishing_point is not None:
            cv2.circle(
                frame,
                (int(result.vanishing_point[0]), int(result.vanishing_point[1])),
                5,
                (0, 255, 255),
                2,
            )
        anchor_y = int(height * 0.90)
        camera_x = int(result.camera_center)
        cv2.line(frame, (camera_x, anchor_y - 18), (camera_x, anchor_y + 18), (255, 80, 0), 2)
        if result.candidate_center is not None:
            cv2.circle(frame, (int(result.candidate_center), anchor_y), 6, (180, 0, 255), 2)
        if result.lane_center is not None:
            cv2.circle(frame, (int(result.lane_center), anchor_y), 6, (0, 255, 255), -1)
        offset = "n/a" if result.offset is None else f"{result.offset:+.2f}"
        candidate_offset = "n/a" if result.candidate_offset is None else f"{result.candidate_offset:+.2f}"
        center = "n/a" if result.lane_center is None else f"{result.lane_center:.1f}"
        if debug:
            reason = result.rejection_reason or "none"
            width_text = "n/a" if result.lane_width_normalized is None else f"{result.lane_width_normalized:.2f}"
            cv2.putText(frame, "ROI yellow raw gray left cyan right magenta", (12, max(20, height - 112)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if result.valid:
                status = "TRUSTED"
            elif result.geometry_valid and reason == "temporal_warmup":
                status = f"GEOMETRY PASS | TRUST WARMUP {result.trusted_count}/{result.trust_required}"
            elif result.geometry_valid:
                status = f"GEOMETRY PASS | REJECT: {reason}"
            else:
                status = f"GEOMETRY REJECT: {reason}"
            cv2.putText(frame, f"{status} conf={result.confidence:.2f}", (12, max(20, height - 88)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0) if result.valid else (0, 200, 255) if result.geometry_valid else (0, 0, 255), 2)
            vp_text = "n/a" if result.vanishing_point is None else f"{result.vanishing_point[0]:.0f},{result.vanishing_point[1]:.0f}"
            cv2.putText(frame, f"support L/R={result.left_support:.0f}/{result.right_support:.0f} mark={result.marking_evidence:.2f} road std={result.surface_texture_std:.1f} width={width_text} vp={vp_text}", (12, max(40, height - 64)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1)
            cv2.putText(frame, f"center={center} candidate offset={candidate_offset} offset={offset} drift={result.drift_state} persistence={result.persistence_count}/{result.persistence_required}", (12, max(60, height - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(frame, f"Normalized offset: {offset}  lane time: {result.processing_ms:.1f} ms", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
def draw_detections(frame: np.ndarray, detections: tuple[Detection, ...]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 0),
            2,
        )
