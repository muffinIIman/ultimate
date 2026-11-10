"""Geometry utilities: polygon validation, point-in-polygon zones and ROI config I/O.

This module has no dependency on any detector or tracker, so it can be unit-tested
in isolation and reused by the ROI selector, the analytics engine and the dashboard.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:  # Shapely is optional: cv2.pointPolygonTest is the fallback backend.
    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.prepared import prep as shapely_prep

    SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)

PointLike = Tuple[float, float]
BoxXYXY = Tuple[float, float, float, float]
ROI_SCHEMA_VERSION = 1


class PolygonError(ValueError):
    """Raised when polygon coordinates are invalid (too few points, NaN, self-crossing...)."""


def polygon_area(points: np.ndarray) -> float:
    """Return the absolute polygon area using the shoelace formula.

    Args:
        points: Array of shape (N, 2).

    Returns:
        Unsigned area in squared pixels.
    """
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the signed cross product (b - a) x (c - a)."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Return True if collinear point b lies within the bounding box of segment a-c."""
    return bool(
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> bool:
    """Return True if segment p1-q1 intersects segment p2-q2 (touching counts)."""
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def is_self_intersecting(points: np.ndarray) -> bool:
    """Check whether a closed polygon has non-adjacent edges that cross or touch.

    Args:
        points: Array of shape (N, 2) with N >= 3.

    Returns:
        True if the polygon is self-intersecting.
    """
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex by definition
            b1, b2 = points[j], points[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def validate_polygon(
    points: Union[Sequence[Sequence[float]], np.ndarray],
    min_vertices: int = 3,
    min_area: float = 1.0,
) -> np.ndarray:
    """Validate polygon coordinates and return them as a float32 (N, 2) array.

    Args:
        points: Iterable of (x, y) vertices.
        min_vertices: Minimum accepted vertex count.
        min_area: Minimum accepted area in squared pixels.

    Returns:
        Validated float32 array of shape (N, 2).

    Raises:
        PolygonError: If the polygon is malformed, degenerate or self-intersecting.
    """
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolygonError(f"Polygon coordinates are not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise PolygonError(f"Polygon must have shape (N, 2), got {arr.shape}.")
    if arr.shape[0] < min_vertices:
        raise PolygonError(f"Polygon needs at least {min_vertices} vertices, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise PolygonError("Polygon contains NaN or infinite coordinates.")
    if polygon_area(arr) < min_area:
        raise PolygonError("Polygon area is (almost) zero: vertices are collinear or duplicated.")
    if is_self_intersecting(arr):
        raise PolygonError("Polygon edges cross each other (self-intersecting polygon).")
    return arr.astype(np.float32)


def bottom_center(box: BoxXYXY) -> PointLike:
    """Return the bottom-center ("feet") point of an (x1, y1, x2, y2) bounding box."""
    x1, _, x2, y2 = box
    return ((float(x1) + float(x2)) / 2.0, float(y2))


class PolygonZone:
    """Point-in-polygon test with a Shapely backend and a cv2 fallback.

    Points exactly on the border are considered inside (`covers` / `pointPolygonTest >= 0`).
    """

    def __init__(self, points: Union[Sequence[Sequence[float]], np.ndarray], use_shapely: bool = True) -> None:
        """Create a zone.

        Args:
            points: Polygon vertices in pixel coordinates.
            use_shapely: Prefer Shapely when installed.

        Raises:
            PolygonError: If the polygon is invalid.
        """
        self._points: np.ndarray = validate_polygon(points)
        self._contour: np.ndarray = self._points.reshape(-1, 1, 2).astype(np.float32)
        self._prepared = None
        self._backend = "cv2"
        if use_shapely and SHAPELY_AVAILABLE:
            try:
                polygon = ShapelyPolygon(self._points)
                if polygon.is_valid:
                    self._prepared = shapely_prep(polygon)
                    self._backend = "shapely"
                else:
                    logger.warning("Shapely rejected the polygon as invalid; using cv2 backend.")
            except Exception as exc:  # noqa: BLE001 - any Shapely failure must degrade gracefully
                logger.warning("Shapely backend failed (%s); using cv2 backend.", exc)

    @property
    def points(self) -> np.ndarray:
        """Return a copy of the polygon vertices as float32 (N, 2)."""
        return self._points.copy()

    @property
    def points_int(self) -> np.ndarray:
        """Return the polygon vertices rounded to int32 (N, 2), ready for cv2 drawing."""
        return np.round(self._points).astype(np.int32)

    @property
    def backend(self) -> str:
        """Return the active point-in-polygon backend name."""
        return self._backend

    def contains(self, point: PointLike) -> bool:
        """Return True if the point lies inside or on the border of the polygon."""
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if self._prepared is not None:
            return bool(self._prepared.covers(ShapelyPoint(x, y)))
        return cv2.pointPolygonTest(self._contour, (x, y), False) >= 0


@dataclass(frozen=True)
class RoiConfig:
    """ROI polygon plus the frame size it was drawn on."""

    polygon: List[List[int]]
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    source: Optional[str] = None

    def scaled_polygon(self, width: int, height: int) -> np.ndarray:
        """Return the polygon rescaled to a (width, height) frame and clipped to its bounds.

        Args:
            width: Target frame width in pixels.
            height: Target frame height in pixels.

        Returns:
            Validated float32 polygon of shape (N, 2).

        Raises:
            PolygonError: If the rescaled polygon is invalid.
        """
        pts = np.asarray(self.polygon, dtype=np.float64)
        if self.frame_width and self.frame_height and width > 0 and height > 0:
            if (self.frame_width, self.frame_height) != (width, height):
                pts = pts * np.array([width / self.frame_width, height / self.frame_height])
        if width > 0 and height > 0 and pts.ndim == 2 and pts.shape[1] == 2:
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        return validate_polygon(pts)


def save_roi(
    path: Union[str, Path],
    points: Sequence[Sequence[float]],
    frame_size: Tuple[int, int],
    source: Optional[str] = None,
) -> Path:
    """Validate and atomically write the ROI polygon to a JSON file.

    Args:
        path: Destination JSON path.
        points: Polygon vertices in pixel coordinates.
        frame_size: (width, height) of the frame the polygon was drawn on.
        source: Optional video path or URL, stored for traceability.

    Returns:
        The written path.

    Raises:
        PolygonError: If the polygon is invalid.
        OSError: If the file cannot be written.
    """
    validated = validate_polygon(points)
    payload = {
        "version": ROI_SCHEMA_VERSION,
        "source": source,
        "frame_width": int(frame_size[0]),
        "frame_height": int(frame_size[1]),
        "polygon": [[int(round(float(x))), int(round(float(y)))] for x, y in validated],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, out)
    return out


def load_roi(path: Union[str, Path]) -> RoiConfig:
    """Load and validate an ROI JSON file.

    Args:
        path: Path to `roi_config.json`.

    Returns:
        Parsed RoiConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        PolygonError: If the content is malformed or the polygon is invalid.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"ROI config not found: {src}. Run roi_selector.py first.")
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolygonError(f"ROI config is not valid JSON: {exc}") from exc
    return parse_roi_dict(data)


def parse_roi_dict(data: dict) -> RoiConfig:
    """Build a RoiConfig from a decoded JSON dictionary.

    Args:
        data: Dictionary with a `polygon` key and optional frame size keys.

    Returns:
        Validated RoiConfig.

    Raises:
        PolygonError: If required keys are missing or the polygon is invalid.
    """
    if not isinstance(data, dict) or "polygon" not in data:
        raise PolygonError("ROI config must be a JSON object with a 'polygon' key.")
    validated = validate_polygon(data["polygon"])
    width = data.get("frame_width")
    height = data.get("frame_height")
    return RoiConfig(
        polygon=[[int(round(float(x))), int(round(float(y)))] for x, y in validated],
        frame_width=int(width) if width else None,
        frame_height=int(height) if height else None,
        source=data.get("source"),
    )
