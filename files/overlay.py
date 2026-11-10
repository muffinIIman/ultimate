"""OpenCV overlay rendering shared by the CLI and the Streamlit dashboard."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from geometry import PolygonZone
from tracker_engine import FrameResult

COLOR_OK = (0, 200, 0)
COLOR_ALERT = (0, 0, 255)
COLOR_IN_QUEUE = (0, 215, 255)
COLOR_OUTSIDE = (200, 200, 200)
COLOR_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS (HH:MM:SS when above one hour)."""
    total = int(max(0.0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def draw_zone(frame: np.ndarray, zone: PolygonZone, color: tuple = COLOR_OK, alpha: float = 0.25) -> np.ndarray:
    """Draw the translucent ROI polygon on a copy of the frame.

    Args:
        frame: BGR image.
        zone: Queue polygon.
        color: BGR colour.
        alpha: Fill opacity.

    Returns:
        Annotated copy.

    Raises:
        ValueError: If the frame is empty.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot draw on an empty frame.")
    canvas = frame.copy()
    pts = zone.points_int
    layer = canvas.copy()
    cv2.fillPoly(layer, [pts], color)
    cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0, dst=canvas)
    cv2.polylines(canvas, [pts], True, color, max(2, canvas.shape[1] // 640), cv2.LINE_AA)
    return canvas


def draw_overlay(
    frame: np.ndarray,
    result: FrameResult,
    zone: PolygonZone,
    fps: Optional[float] = None,
) -> np.ndarray:
    """Render polygon, boxes, IDs, live counters and the alert banner.

    Args:
        frame: Original BGR frame (not modified).
        result: Output of `QueueAnalyzer.process_frame` for this frame.
        zone: Queue polygon.
        fps: Optional processing FPS to display.

    Returns:
        New annotated BGR frame.

    Raises:
        ValueError: If the frame is empty.
    """
    metrics = result.metrics
    alert = metrics.alert.active
    canvas = draw_zone(frame, zone, COLOR_ALERT if alert else COLOR_OK)
    height, width = canvas.shape[:2]
    scale = max(0.5, width / 1280.0)
    thickness = max(1, int(round(2 * scale)))

    for person in result.people:
        x1, y1, x2, y2 = person.box
        color = COLOR_IN_QUEUE if person.in_roi else COLOR_OUTSIDE
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        fx, fy = int(round(person.feet[0])), int(round(person.feet[1]))
        cv2.circle(canvas, (fx, fy), max(3, int(4 * scale)), (255, 0, 0), -1, cv2.LINE_AA)
        label = f"ID {person.track_id}"
        if person.in_roi:
            label += f" | {format_seconds(person.dwell_seconds)}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5 * scale, 1)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(canvas, label, (x1 + 3, top + th + 1), FONT, 0.5 * scale, (0, 0, 0), 1, cv2.LINE_AA)

    lines: List[str] = [
        f"Queue length : {metrics.queue_length}",
        f"Avg wait     : {format_seconds(metrics.avg_wait_seconds)}",
        f"Current dwell: {format_seconds(metrics.current_avg_dwell_seconds)}",
        f"Longest wait : {format_seconds(metrics.max_dwell_seconds)}",
        f"Entries/Exits: {metrics.total_entries}/{metrics.total_exits}",
        f"Time         : {format_seconds(metrics.timestamp)}",
    ]
    if fps is not None:
        lines.append(f"FPS          : {fps:.1f}")
    line_height = int(24 * scale)
    panel_w = int(300 * scale)
    panel_top = int(60 * scale) if alert else 10
    panel_h = line_height * len(lines) + 12
    layer = canvas.copy()
    cv2.rectangle(layer, (10, panel_top), (10 + panel_w, panel_top + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(layer, 0.55, canvas, 0.45, 0, dst=canvas)
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (18, panel_top + line_height * (index + 1)),
            cv2.FONT_HERSHEY_PLAIN,
            1.1 * scale,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    if alert:
        reasons = ", ".join(metrics.alert.reasons) or "threshold exceeded"
        banner = f"CONGESTION ALERT: {reasons}"
        cv2.rectangle(canvas, (0, 0), (width, int(48 * scale)), COLOR_ALERT, -1)
        font_scale = 0.8 * scale
        (tw, _), _ = cv2.getTextSize(banner, FONT, font_scale, 2)
        while tw > width - 20 and font_scale > 0.3:
            font_scale -= 0.05
            (tw, _), _ = cv2.getTextSize(banner, FONT, font_scale, 2)
        cv2.putText(canvas, banner, (10, int(32 * scale)), FONT, font_scale, COLOR_TEXT, 2, cv2.LINE_AA)
    return canvas
