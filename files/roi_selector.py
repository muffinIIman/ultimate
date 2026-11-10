"""Interactive Queue-polygon ROI selector.

Usage:
    python roi_selector.py --source data/cafeteria.mp4 --output roi_config.json
    python roi_selector.py --source rtsp://user:pass@host/stream --frame-index 0

Controls:
    Left click   add a vertex
    Right click / U / Backspace   undo last vertex
    R            reset all vertices
    Enter / S    validate and save (needs >= 4 vertices)
    Q / Esc      cancel without saving
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from geometry import PolygonError, is_self_intersecting, save_roi, validate_polygon
from video_io import grab_frame

logger = logging.getLogger(__name__)

WINDOW_NAME = "ROI Selector - queue polygon"
MIN_VERTICES = 4
DUPLICATE_CLICK_PIXELS = 3

COLOR_VALID = (0, 200, 0)
COLOR_INVALID = (0, 0, 255)
COLOR_PENDING = (0, 215, 255)
COLOR_TEXT = (255, 255, 255)


class RoiSelector:
    """Mouse-driven polygon editor operating in original-frame coordinates."""

    def __init__(self, frame: np.ndarray, min_vertices: int = MIN_VERTICES, max_display_width: int = 1280) -> None:
        """Prepare the editor.

        Args:
            frame: BGR frame to draw on.
            min_vertices: Minimum vertices required to accept the polygon.
            max_display_width: Frames wider than this are scaled down for display only.

        Raises:
            ValueError: If the frame is empty or not a 3-channel image.
        """
        if frame is None or frame.size == 0 or frame.ndim != 3:
            raise ValueError("RoiSelector needs a non-empty BGR frame.")
        self._frame = frame
        self._height, self._width = frame.shape[:2]
        self._min_vertices = max(3, min_vertices)
        self._scale = min(1.0, max_display_width / float(self._width))
        if self._scale < 1.0:
            size = (int(self._width * self._scale), int(self._height * self._scale))
            self._base = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            self._base = frame.copy()
        self._points: List[Tuple[int, int]] = []
        self._cursor: Optional[Tuple[int, int]] = None
        self._message: str = ""

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of the original frame."""
        return self._width, self._height

    # ------------------------------------------------------------------ coordinates
    def _to_frame(self, x: int, y: int) -> Tuple[int, int]:
        """Convert display coordinates to clamped original-frame coordinates."""
        fx = int(round(x / self._scale))
        fy = int(round(y / self._scale))
        return min(max(fx, 0), self._width - 1), min(max(fy, 0), self._height - 1)

    def _to_display(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Convert original-frame coordinates to display coordinates."""
        return int(round(point[0] * self._scale)), int(round(point[1] * self._scale))

    # ------------------------------------------------------------------ events
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV mouse callback."""
        point = self._to_frame(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self._cursor = point
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self._points and np.hypot(point[0] - self._points[-1][0], point[1] - self._points[-1][1]) < DUPLICATE_CLICK_PIXELS:
                self._message = "Ignored duplicate click."
                return
            self._points.append(point)
            self._message = ""
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._undo()

    def _undo(self) -> None:
        """Remove the last vertex if any."""
        if self._points:
            self._points.pop()
            self._message = "Last vertex removed."

    def _reset(self) -> None:
        """Remove all vertices."""
        self._points.clear()
        self._message = "Polygon reset."

    # ------------------------------------------------------------------ validation
    def _current_status(self) -> Tuple[Tuple[int, int, int], str]:
        """Return (draw colour, status text) for the current vertex set."""
        n = len(self._points)
        if n < 3:
            return COLOR_PENDING, f"{n} vertices - need at least {self._min_vertices}"
        if is_self_intersecting(np.asarray(self._points, dtype=np.float64)):
            return COLOR_INVALID, "INVALID: edges cross - undo or reset"
        if n < self._min_vertices:
            return COLOR_PENDING, f"{n} vertices - need at least {self._min_vertices}"
        return COLOR_VALID, f"{n} vertices - valid, press Enter to save"

    def _try_accept(self) -> Optional[np.ndarray]:
        """Validate the polygon; return it when acceptable, else set an error message."""
        if len(self._points) < self._min_vertices:
            self._message = f"Add at least {self._min_vertices} vertices first."
            return None
        try:
            return validate_polygon(self._points, min_vertices=self._min_vertices)
        except PolygonError as exc:
            self._message = str(exc)
            return None

    # ------------------------------------------------------------------ rendering
    def _render(self) -> np.ndarray:
        """Draw the current state on a copy of the display frame."""
        canvas = self._base.copy()
        color, status = self._current_status()
        display_pts = [self._to_display(p) for p in self._points]
        if len(display_pts) >= 3:
            layer = canvas.copy()
            cv2.fillPoly(layer, [np.asarray(display_pts, dtype=np.int32)], color)
            cv2.addWeighted(layer, 0.25, canvas, 0.75, 0, dst=canvas)
            cv2.polylines(canvas, [np.asarray(display_pts, dtype=np.int32)], True, color, 2, cv2.LINE_AA)
        elif len(display_pts) == 2:
            cv2.line(canvas, display_pts[0], display_pts[1], color, 2, cv2.LINE_AA)
        if display_pts and self._cursor is not None:  # rubber band to the mouse
            cv2.line(canvas, display_pts[-1], self._to_display(self._cursor), COLOR_PENDING, 1, cv2.LINE_AA)
        for index, pt in enumerate(display_pts, start=1):
            cv2.circle(canvas, pt, 5, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)
        lines = [
            "L-click: add | R-click/U: undo | R: reset | Enter/S: save | Q/Esc: cancel",
            status,
        ]
        if self._message:
            lines.append(self._message)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 10 + 24 * len(lines)), (0, 0, 0), -1)
        for row, text in enumerate(lines):
            tint = color if row == 1 else COLOR_TEXT
            cv2.putText(canvas, text, (10, 24 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tint, 1, cv2.LINE_AA)
        return canvas

    # ------------------------------------------------------------------ main loop
    def run(self) -> Optional[np.ndarray]:
        """Run the interactive loop.

        Returns:
            Validated polygon (N, 2) float32, or None if the user cancelled.

        Raises:
            cv2.error: If no display is available (e.g. headless OpenCV build).
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self._render())
                key = cv2.waitKey(20) & 0xFF
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    return None
                if key in (27, ord("q")):
                    return None
                if key in (ord("u"), 8, ord("z")):
                    self._undo()
                elif key == ord("r"):
                    self._reset()
                elif key in (13, 10, ord("s"), ord("c")):
                    polygon = self._try_accept()
                    if polygon is not None:
                        return polygon
        finally:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Define the queue polygon ROI on a video frame.")
    parser.add_argument("--source", required=True, help="Video file, image, RTSP URL or webcam index.")
    parser.add_argument("--output", default="roi_config.json", help="Output JSON path (default: roi_config.json).")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame to draw on (video files only).")
    parser.add_argument("--min-vertices", type=int, default=MIN_VERTICES, help="Minimum polygon vertices.")
    parser.add_argument("--max-display-width", type=int, default=1280, help="Scale large frames down for display.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code (0 on success, 1 on cancel, 2 on error).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        frame = grab_frame(args.source, args.frame_index)
        selector = RoiSelector(frame, min_vertices=args.min_vertices, max_display_width=args.max_display_width)
        polygon = selector.run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except cv2.error as exc:
        logger.error("OpenCV GUI unavailable (headless build?): %s", exc)
        return 2
    if polygon is None:
        logger.info("Cancelled - nothing saved.")
        return 1
    try:
        written = save_roi(args.output, polygon, selector.frame_size, source=str(args.source))
    except (PolygonError, OSError) as exc:
        logger.error("Could not save ROI: %s", exc)
        return 2
    logger.info("Saved %d-vertex polygon to %s", len(polygon), written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
