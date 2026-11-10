"""Frame ingestion (video files, webcams, RTSP) and annotated video writing."""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from types import TracebackType
from typing import Optional, Tuple, Type, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
DEFAULT_FPS = 25.0


def parse_source(source: Union[str, int]) -> Union[str, int]:
    """Normalise a user-provided source (digit strings become webcam indices).

    Args:
        source: File path, URL or webcam index.

    Returns:
        An int webcam index or a stripped string.

    Raises:
        ValueError: If the source is empty.
    """
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if not text:
        raise ValueError("Video source must not be empty.")
    return int(text) if text.isdigit() else text


def is_live_source(source: Union[str, int]) -> bool:
    """Return True for webcams and network streams (no end, may need reconnecting)."""
    if isinstance(source, int):
        return True
    return str(source).lower().startswith(LIVE_PREFIXES)


class FrameSource:
    """Robust frame reader with video-time timestamps and automatic RTSP reconnection.

    Timestamps are derived from the frame index for files (reproducible results, independent
    of processing speed) and from a monotonic clock for live sources.
    """

    def __init__(
        self,
        source: Union[str, int],
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        max_bad_file_frames: int = 3,
    ) -> None:
        """Configure the reader (call `open()` or use it as a context manager).

        Args:
            source: File path, RTSP/HTTP URL or webcam index.
            reconnect_attempts: Reconnection attempts for live sources.
            reconnect_delay: Seconds to wait between attempts.
            max_bad_file_frames: Consecutive undecodable frames tolerated in a file.
        """
        self._source = parse_source(source)
        self._live = is_live_source(self._source)
        self._reconnect_attempts = max(0, reconnect_attempts)
        self._reconnect_delay = max(0.0, reconnect_delay)
        self._max_bad_file_frames = max(1, max_bad_file_frames)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = DEFAULT_FPS
        self._width: int = 0
        self._height: int = 0
        self._frame_count: int = 0
        self._frame_index: int = 0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def fps(self) -> float:
        """Nominal frames per second (falls back to 25 when metadata is missing)."""
        return self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def frame_count(self) -> int:
        """Total frames for files; 0 when unknown (live sources)."""
        return self._frame_count

    @property
    def is_live(self) -> bool:
        """True for webcams and network streams."""
        return self._live

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> "FrameSource":
        """Open the capture device.

        Returns:
            self, for chaining.

        Raises:
            FileNotFoundError: If a local file does not exist.
            RuntimeError: If the capture cannot be opened.
        """
        if isinstance(self._source, str) and not self._live and not Path(self._source).is_file():
            raise FileNotFoundError(f"Video file not found: {self._source}")
        if isinstance(self._source, str) and self._source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video source: {self._source!r}")
        if self._live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low; ignored by some backends
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 1e-3 or fps > 240.0:
            logger.warning("Invalid FPS metadata (%s); assuming %.0f FPS.", fps, DEFAULT_FPS)
            fps = DEFAULT_FPS
        self._cap = cap
        self._fps = fps
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_count = max(0, count) if not self._live else 0
        self._frame_index = 0
        self._start_time = time.monotonic()
        return self

    def release(self) -> None:
        """Release the capture device (safe to call repeatedly)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()

    # ------------------------------------------------------------------ reading
    def _reconnect(self) -> bool:
        """Try to reopen a live source. Returns True on success."""
        self.release()
        for attempt in range(1, self._reconnect_attempts + 1):
            logger.warning("Stream lost. Reconnecting (%d/%d)...", attempt, self._reconnect_attempts)
            time.sleep(self._reconnect_delay)
            try:
                self.open()
                # open() resets the timeline; keep the monotonic start for continuity.
                return True
            except (RuntimeError, FileNotFoundError):
                continue
        logger.error("Could not reconnect to %r after %d attempts.", self._source, self._reconnect_attempts)
        return False

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """Read the next valid frame.

        Returns:
            (frame_bgr, timestamp_seconds), or None at end of stream / unrecoverable failure.
        """
        bad_frames = 0
        while True:
            if self._cap is None or not self._cap.isOpened():
                if not (self._live and self._reconnect()):
                    return None
            assert self._cap is not None
            try:
                ok, frame = self._cap.read()
            except cv2.error as exc:
                logger.warning("cv2 read error: %s", exc)
                ok, frame = False, None
            if ok and frame is not None and frame.size > 0:
                timestamp = (
                    time.monotonic() - self._start_time if self._live else self._frame_index / self._fps
                )
                self._frame_index += 1
                return frame, float(timestamp)
            bad_frames += 1
            if self._live:
                if not self._reconnect():
                    return None
            elif bad_frames >= self._max_bad_file_frames:
                return None  # end of file (or a truncated/corrupt tail)


def grab_frame(source: Union[str, int], frame_index: int = 0) -> np.ndarray:
    """Return one frame from an image, a video file or a stream.

    Args:
        source: Image path, video path, URL or webcam index.
        frame_index: Index of the frame to extract (ignored for images and live sources).

    Returns:
        BGR frame.

    Raises:
        RuntimeError: If no frame can be read.
    """
    parsed = parse_source(source)
    if isinstance(parsed, str) and Path(parsed).suffix.lower() in IMAGE_EXTENSIONS:
        image = cv2.imread(parsed)
        if image is None:
            raise RuntimeError(f"Cannot read image: {parsed}")
        return image
    with FrameSource(parsed) as reader:
        target = 0 if reader.is_live else max(0, frame_index)
        frame: Optional[np.ndarray] = None
        for _ in range(target + 1):
            item = reader.read()
            if item is None:
                break
            frame = item[0]
        if frame is None:
            raise RuntimeError(f"Could not read a frame from {parsed!r}.")
        return frame


class VideoSink:
    """Annotated video writer with codec fallback and frame-size safety."""

    def __init__(self, path: Union[str, Path], fps: float, frame_size: Tuple[int, int]) -> None:
        """Open the output file.

        Args:
            path: Output path (parent folders are created).
            fps: Output frame rate.
            frame_size: (width, height).

        Raises:
            RuntimeError: If no codec can open the file.
            ValueError: If the frame size is invalid.
        """
        width, height = int(frame_size[0]), int(frame_size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame size: {frame_size}")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._size = (width, height)
        safe_fps = fps if math.isfinite(fps) and fps > 0 else DEFAULT_FPS
        self._writer: Optional[cv2.VideoWriter] = None
        for codec in ("mp4v", "avc1", "XVID"):
            writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*codec), safe_fps, self._size)
            if writer.isOpened():
                self._writer = writer
                logger.info("Writing %s with codec %s.", self._path, codec)
                break
            writer.release()
        if self._writer is None:
            raise RuntimeError(f"No usable video codec to write {self._path}.")

    @property
    def path(self) -> Path:
        """Output file path."""
        return self._path

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR frame (resized if its size differs from the sink size)."""
        if self._writer is None or frame is None or frame.size == 0:
            return
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def release(self) -> None:
        """Flush and close the writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()
