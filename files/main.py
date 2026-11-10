"""CLI entry point for the Intelligent Queue & Waiting Time Analyzer.

Examples:
    python main.py --source data/cafeteria.mp4 --roi roi_config.json --output outputs/annotated.mp4
    python main.py --source rtsp://user:pass@192.168.1.10/stream --roi roi_config.json --max-queue 10 --max-wait 180
    python main.py --source data/desk.mp4 --roi roi_config.json --no-show --metrics-csv outputs/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO

import cv2

from geometry import PolygonError, PolygonZone, load_roi
from overlay import draw_overlay
from tracker_engine import AlertConfig, AnalyzerConfig, QueueAnalyzer
from video_io import FrameSource, VideoSink

logger = logging.getLogger("queue_analyzer")

METRICS_FIELDS = (
    "frame_index",
    "timestamp",
    "queue_length",
    "avg_wait_seconds",
    "current_avg_dwell_seconds",
    "max_dwell_seconds",
    "effective_wait_seconds",
    "total_entries",
    "total_exits",
    "alert_active",
    "inference_ms",
)
MAX_CONSECUTIVE_ERRORS = 10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Queue length and waiting-time analytics (YOLOv8 + ByteTrack).")
    p.add_argument("--source", required=True, help="Video file, RTSP/HTTP URL or webcam index.")
    p.add_argument("--roi", default="roi_config.json", help="ROI JSON produced by roi_selector.py.")
    p.add_argument("--output", default="outputs/annotated.mp4", help="Annotated video path ('' to disable).")
    p.add_argument("--metrics-csv", default="", help="Optional per-frame metrics CSV path.")
    p.add_argument("--event-log", default="", help="Optional completed-visits CSV path.")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights (yolov8n.pt / yolov8s.pt).")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (bytetrack.yaml / botsort.yaml).")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--no-half", action="store_true", help="Disable FP16 on CUDA.")
    p.add_argument("--no-shapely", action="store_true", help="Use cv2.pointPolygonTest instead of Shapely.")
    p.add_argument("--exit-grace", type=float, default=1.5, help="Seconds a track may vanish before exit.")
    p.add_argument("--min-wait", type=float, default=1.0, help="Ignore visits shorter than this (seconds).")
    p.add_argument("--wait-window", type=int, default=50, help="Completed waits in the rolling average.")
    p.add_argument("--max-queue", type=int, default=8, help="Alert if queue length exceeds this value.")
    p.add_argument("--max-wait", type=float, default=120.0, help="Alert if average wait (s) exceeds this value.")
    p.add_argument("--consecutive-frames", type=int, default=15, help="Frames the condition must persist (M).")
    p.add_argument("--frame-stride", type=int, default=1, help="Process every k-th frame.")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames (0 = no limit).")
    p.add_argument("--no-show", action="store_true", help="Do not open a preview window.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def build_analyzer(args: argparse.Namespace, zone: PolygonZone) -> QueueAnalyzer:
    """Create the analyzer from parsed arguments.

    Args:
        args: Parsed CLI arguments.
        zone: Queue polygon.

    Returns:
        Ready-to-use QueueAnalyzer.
    """
    config = AnalyzerConfig(
        model_path=args.model,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=not args.no_half,
        exit_grace_seconds=args.exit_grace,
        min_wait_seconds=args.min_wait,
        wait_window=args.wait_window,
        use_shapely=not args.no_shapely,
    )
    alerts = AlertConfig(
        max_queue_length=args.max_queue,
        max_wait_seconds=args.max_wait,
        consecutive_frames=args.consecutive_frames,
    )
    return QueueAnalyzer(zone, config, alerts)


def _open_metrics_writer(path: str) -> tuple[Optional[TextIO], Optional[Any]]:
    """Open the per-frame metrics CSV (returns (handle, writer) or (None, None))."""
    if not path:
        return None, None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(METRICS_FIELDS)
    return handle, writer


def run(args: argparse.Namespace) -> int:
    """Execute the analysis loop.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 success, 2 configuration/runtime error).
    """
    if args.frame_stride < 1:
        logger.error("--frame-stride must be >= 1.")
        return 2
    try:
        roi = load_roi(args.roi)
    except (FileNotFoundError, PolygonError) as exc:
        logger.error("%s", exc)
        return 2

    show = not args.no_show
    sink: Optional[VideoSink] = None
    metrics_handle: Optional[TextIO] = None
    metrics_writer: Optional[Any] = None
    processed = 0
    alert_frames = 0
    inference_total = 0.0
    consecutive_errors = 0
    smoothed_fps = 0.0
    loop_started = time.perf_counter()
    analyzer: Optional[QueueAnalyzer] = None

    try:
        with FrameSource(args.source) as source:
            if source.width <= 0 or source.height <= 0:
                raise RuntimeError("Video source did not report a valid frame size.")
            polygon = roi.scaled_polygon(source.width, source.height)
            zone = PolygonZone(polygon, use_shapely=not args.no_shapely)
            analyzer = build_analyzer(args, zone)
            logger.info(
                "Source %dx%d @ %.1f FPS | device=%s | point-in-polygon=%s",
                source.width, source.height, source.fps, analyzer.device, zone.backend,
            )
            if args.output:
                sink = VideoSink(args.output, source.fps / args.frame_stride, (source.width, source.height))
            metrics_handle, metrics_writer = _open_metrics_writer(args.metrics_csv)

            frame_index = -1
            while True:
                item = source.read()
                if item is None:
                    logger.info("End of stream.")
                    break
                frame, timestamp = item
                frame_index += 1
                if frame_index % args.frame_stride:
                    continue

                tick = time.perf_counter()
                try:
                    result = analyzer.process_frame(frame, timestamp)
                    consecutive_errors = 0
                except RuntimeError as exc:
                    consecutive_errors += 1
                    logger.error("Frame %d failed (%d/%d): %s", frame_index, consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error("Too many consecutive failures; aborting.")
                        return 2
                    continue

                elapsed = max(time.perf_counter() - tick, 1e-6)
                instant_fps = 1.0 / elapsed
                smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
                processed += 1
                inference_total += result.inference_ms
                alert_frames += int(result.metrics.alert.active)

                annotated = draw_overlay(frame, result, zone, fps=smoothed_fps)
                if sink is not None:
                    sink.write(annotated)
                if metrics_writer is not None:
                    m = result.metrics
                    metrics_writer.writerow(
                        [frame_index, f"{m.timestamp:.3f}", m.queue_length, f"{m.avg_wait_seconds:.3f}",
                         f"{m.current_avg_dwell_seconds:.3f}", f"{m.max_dwell_seconds:.3f}",
                         f"{m.effective_wait_seconds:.3f}", m.total_entries, m.total_exits,
                         int(m.alert.active), f"{result.inference_ms:.2f}"]
                    )
                if show:
                    try:
                        cv2.imshow("Queue Analyzer (press q to quit)", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("Quit requested.")
                            break
                    except cv2.error as exc:
                        logger.warning("Preview disabled (no GUI available): %s", exc)
                        show = False
                if args.max_frames and processed >= args.max_frames:
                    logger.info("Reached --max-frames=%d.", args.max_frames)
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except (FileNotFoundError, RuntimeError, PolygonError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    finally:
        if sink is not None:
            sink.release()
        if metrics_handle is not None:
            metrics_handle.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    wall = max(time.perf_counter() - loop_started, 1e-6)
    if analyzer is not None and processed:
        if args.event_log:
            logger.info("Event log written to %s", analyzer.save_event_log_csv(args.event_log))
        last = analyzer.get_event_log()
        logger.info(
            "Done: %d frames | %.1f FPS end-to-end | %.1f ms mean inference | alert frames: %d | completed visits: %d",
            processed, processed / wall, inference_total / processed, alert_frames, len(last),
        )
        if args.output:
            logger.info("Annotated video: %s", args.output)
    return 0


def main() -> None:
    """Parse arguments, configure logging and exit with the run status."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
