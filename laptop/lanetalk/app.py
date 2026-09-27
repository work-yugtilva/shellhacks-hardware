"""Display the live ESP32-CAM stream with local driving-event detections."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import time

import cv2
import numpy as np

from lanetalk.camera import CameraStream
from lanetalk.coach import CoachWorker
from lanetalk.detector import (
    LaneDetector,
    LaneValidityConfig,
    ObjectDetector,
    ObjectDetectorSnapshot,
    draw_detections,
)
from lanetalk.events import Event


def _event_text(event: Event) -> str:
    if event.kind == "lane_drift":
        direction = event.details.get("direction", "unknown")
        return f"[LANE] drift {direction} confidence={event.confidence:.2f}"
    class_name = event.details.get("class", "object")
    return f"[OBJECT] {class_name} confidence={event.confidence:.2f}"


def _draw_text(
    frame: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaneTalk local camera detections")
    parser.add_argument(
        "--lane-debug",
        action="store_true",
        help="overlay ROI, Hough segments, accepted lane candidates, and lane state",
    )
    parser.add_argument(
        "--lane-roi",
        type=_parse_lane_roi,
        metavar="X1,Y1,X2,Y2,X3,Y3,X4,Y4",
        help=(
            "override the generic normalized lane ROI polygon (vertices ordered "
            "bottom-left, top-left, top-right, bottom-right)"
        ),
    )
    return parser.parse_args(argv)


def _parse_lane_roi(value: str) -> tuple[tuple[float, float], ...]:
    try:
        coordinates = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lane ROI must contain eight normalized numbers") from exc
    if len(coordinates) != 8:
        raise argparse.ArgumentTypeError("lane ROI must contain eight normalized numbers")
    vertices = tuple(zip(coordinates[::2], coordinates[1::2]))
    try:
        LaneValidityConfig(roi_vertices=vertices)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return vertices


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    lane_debug = args.lane_debug
    window = "LaneTalk Phase 2"
    object_detector = ObjectDetector()
    lane_config = LaneValidityConfig(roi_vertices=args.lane_roi) if args.lane_roi else LaneValidityConfig()
    lane_detector = LaneDetector(config=lane_config)
    coach = CoachWorker()
    frame_count = 0
    fps = 0.0
    frame_period_ms = 1000.0 / 30.0
    previous_frame_at: float | None = None
    loop_time_total_ms = 0.0
    lane_time_total_ms = 0.0
    lane_sample_count = 0
    inference_samples: list[float] = []
    last_snapshot: ObjectDetectorSnapshot | None = None
    visible_events: deque[Event] = deque()
    last_camera_error = ""
    last_detector_error = ""
    saw_object_detection = False
    object_event_count = 0
    lane_event_count = 0
    trusted_lane_frames = 0
    invalid_lane_frames = 0
    trusted_offsets: list[float] = []
    lane_rejection_reasons: Counter[str] = Counter()
    camera_reconnect_count = 0
    first_frame_at: float | None = None

    if lane_debug:
        print(
            "[lane-debug] ROI=yellow raw Hough=gray left candidates=cyan "
            "right candidates=magenta candidate fits=blue/orange trusted fits=green/orange",
            flush=True,
        )

    try:
        object_detector.start()
        with CameraStream() as camera:
            try:
                while True:
                    loop_started = time.perf_counter()
                    frame = camera.read()
                    frame_received = frame is not None

                    if camera.last_error and camera.last_error != last_camera_error:
                        print(f"[camera] {camera.last_error}", flush=True)
                        last_camera_error = camera.last_error
                    elif not camera.last_error:
                        last_camera_error = ""

                    lane_result = None
                    if frame is None:
                        lane_detector.reset()
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            frame,
                            "Connecting to ESP32-CAM...",
                            (24, 185),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        now = time.monotonic()
                        frame_count += 1
                        if first_frame_at is None:
                            first_frame_at = now
                        if previous_frame_at is not None:
                            interval = now - previous_frame_at
                            if interval > 0:
                                sample_period_ms = interval * 1000
                                frame_period_ms = 0.8 * frame_period_ms + 0.2 * sample_period_ms
                                measured_fps = 1.0 / interval
                                fps = measured_fps if fps == 0 else 0.8 * fps + 0.2 * measured_fps
                        previous_frame_at = now

                        lane_result = lane_detector.process(frame)
                        lane_time_total_ms += lane_result.processing_ms
                        lane_sample_count += 1
                        if lane_result.valid:
                            trusted_lane_frames += 1
                            if lane_result.offset is not None:
                                trusted_offsets.append(lane_result.offset)
                        else:
                            invalid_lane_frames += 1
                            if lane_result.rejection_reason:
                                lane_rejection_reasons[lane_result.rejection_reason] += 1
                        for event in lane_result.events:
                            print(_event_text(event), flush=True)
                            visible_events.append(event)
                            coach.submit(event)
                            lane_event_count += 1

                        object_detector.submit(frame, frame_count, frame_period_ms)

                    snapshot = object_detector.snapshot()
                    if snapshot is not last_snapshot:
                        if last_snapshot is not None and snapshot.last_inference_ms > 0:
                            inference_samples.append(snapshot.last_inference_ms)
                        last_snapshot = snapshot
                    if snapshot.error and snapshot.error != last_detector_error:
                        print(f"[object] detector error: {snapshot.error}", flush=True)
                        last_detector_error = snapshot.error

                    for event in object_detector.drain_events():
                        print(_event_text(event), flush=True)
                        visible_events.append(event)
                        coach.submit(event)
                        object_event_count += 1

                    now_wall = time.time()
                    while visible_events and now_wall - visible_events[0].timestamp > 5.0:
                        visible_events.popleft()

                    if frame_received and lane_result is not None:
                        if lane_debug:
                            LaneDetector.draw(frame, lane_result, debug=True)
                        else:
                            LaneDetector.draw(frame, lane_result)
                        draw_detections(frame, snapshot.detections)
                        saw_object_detection = saw_object_detection or bool(snapshot.detections)

                    loop_time_ms = (time.perf_counter() - loop_started) * 1000
                    if frame_received:
                        loop_time_total_ms += loop_time_ms
                    _draw_text(frame, f"Camera FPS: {fps:.1f}  read+process: {loop_time_ms:.1f} ms", 25)
                    yolo_time = (
                        f"{snapshot.average_inference_ms:.1f} ms avg"
                        if snapshot.average_inference_ms > 0
                        else "waiting"
                    )
                    _draw_text(
                        frame,
                        f"YOLO: {snapshot.status}  {yolo_time}  every {snapshot.stride} frames ({snapshot.device})",
                        50,
                        (255, 200, 0),
                    )
                    _draw_text(
                        frame,
                        f"Active detections: {len(snapshot.detections)}",
                        75,
                        (255, 200, 0),
                    )
                    if lane_result is None:
                        _draw_text(frame, "Lane time: --  offset: --", 100, (0, 255, 255))
                    for index, event in enumerate(reversed(visible_events)):
                        if index == 3:
                            break
                        _draw_text(frame, _event_text(event), 128 + index * 22, (255, 255, 255))

                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except KeyboardInterrupt:
                pass
            finally:
                camera_reconnect_count = getattr(camera, "reconnect_count", 0)
                camera.close()
    finally:
        object_detector.close()
        for event in object_detector.drain_events():
            coach.submit(event)
        coach.close()
        cv2.destroyAllWindows()

    object_snapshot = object_detector.snapshot()
    if frame_count:
        average_lane_ms = lane_time_total_ms / lane_sample_count if lane_sample_count else 0.0
        average_loop_ms = loop_time_total_ms / frame_count
        average_camera_fps = 0.0
        if first_frame_at is not None and previous_frame_at is not None and previous_frame_at > first_frame_at:
            average_camera_fps = (frame_count - 1) / (previous_frame_at - first_frame_at)
        average_inference_ms = (
            sum(inference_samples) / len(inference_samples)
            if inference_samples
            else object_snapshot.average_inference_ms
        )
        print(
            f"[metrics] frames={frame_count} camera_fps={fps:.1f} "
            f"session_camera_fps={average_camera_fps:.1f} "
            f"mean_read_process_ms={average_loop_ms:.1f} "
            f"mean_yolo_inference_ms={average_inference_ms:.1f} "
            f"mean_lane_ms={average_lane_ms:.2f}",
            flush=True,
        )
        print(
            f"[metrics] yolo_detections={saw_object_detection} "
            f"lane_trusted={trusted_lane_frames} lane_invalid={invalid_lane_frames} "
            f"object_events={object_event_count} lane_events={lane_event_count} "
            f"camera_reconnects={camera_reconnect_count}",
            flush=True,
        )
        if trusted_offsets:
            offsets = np.asarray(trusted_offsets, dtype=float)
            print(
                f"[metrics] trusted_offset_mean={float(np.mean(offsets)):+.3f} "
                f"median={float(np.median(offsets)):+.3f} std={float(np.std(offsets)):.3f}",
                flush=True,
            )
        print(f"[metrics] lane_rejections={dict(lane_rejection_reasons)}", flush=True)
    else:
        print("[metrics] no camera frames received", flush=True)


if __name__ == "__main__":
    main()
