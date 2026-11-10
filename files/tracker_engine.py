"""Core analytics engine: YOLOv8 + ByteTrack, ROI membership, wait times, congestion alerts.

Data flow inside `QueueAnalyzer.process_frame`:
    frame -> YOLOv8 person detection + ByteTrack IDs -> feet point (bottom-center)
          -> point-in-polygon -> per-ID entry/exit state -> rolling wait metrics
          -> CongestionAlertEngine -> FrameResult
"""

from __future__ import annotations

import csv
import gc
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from geometry import PolygonZone, bottom_center

logger = logging.getLogger(__name__)

EVENT_LOG_FIELDS = ("track_id", "entry_ts", "exit_ts", "wait_seconds")
MAX_EVENT_LOG = 100_000


# ====================================================================== configuration
@dataclass
class AlertConfig:
    """Congestion alert thresholds.

    An alert fires when queue length > `max_queue_length` OR the wait estimate
    > `max_wait_seconds` for `consecutive_frames` consecutive processed frames.
    """

    max_queue_length: int = 8
    max_wait_seconds: float = 120.0
    consecutive_frames: int = 15

    def validate(self) -> None:
        """Raise ValueError if a threshold is out of range."""
        if self.max_queue_length < 0:
            raise ValueError("max_queue_length must be >= 0.")
        if self.max_wait_seconds < 0 or not math.isfinite(self.max_wait_seconds):
            raise ValueError("max_wait_seconds must be a finite value >= 0.")
        if self.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1.")


@dataclass
class AnalyzerConfig:
    """Detector, tracker and wait-time settings."""

    model_path: str = "yolov8n.pt"
    tracker: str = "bytetrack.yaml"  # "botsort.yaml" or a custom tracker YAML also work
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"  # "auto" | "cpu" | "cuda:0" | "mps"
    half: bool = True  # FP16, only applied on CUDA
    person_class_id: int = 0
    exit_grace_seconds: float = 1.5  # absence tolerated before a track is considered gone
    min_wait_seconds: float = 1.0  # shorter visits are treated as flicker, not waits
    wait_window: int = 50  # number of completed waits in the rolling average
    use_shapely: bool = True

    def validate(self) -> None:
        """Raise ValueError if a setting is out of range."""
        if not 0.0 < self.conf <= 1.0:
            raise ValueError("conf must be in (0, 1].")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("iou must be in (0, 1].")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32.")
        if self.exit_grace_seconds < 0 or self.min_wait_seconds < 0:
            raise ValueError("exit_grace_seconds and min_wait_seconds must be >= 0.")
        if self.wait_window < 1:
            raise ValueError("wait_window must be >= 1.")


# ====================================================================== result types
class Detection(NamedTuple):
    """One tracked person detection."""

    box: Tuple[float, float, float, float]
    confidence: float
    track_id: int


@dataclass(frozen=True)
class TrackedPerson:
    """A tracked person in the current frame."""

    track_id: int
    box: Tuple[int, int, int, int]
    confidence: float
    feet: Tuple[float, float]
    in_roi: bool
    dwell_seconds: float


@dataclass(frozen=True)
class AlertStatus:
    """State of the congestion alert for one frame."""

    active: bool = False
    queue_triggered: bool = False
    wait_triggered: bool = False
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameMetrics:
    """Queue metrics for one processed frame."""

    timestamp: float
    queue_length: int
    avg_wait_seconds: float  # rolling mean of COMPLETED waits (0.0 when none yet)
    current_avg_dwell_seconds: float  # mean dwell of people currently queued
    max_dwell_seconds: float  # longest current dwell
    effective_wait_seconds: float  # value compared against the alert threshold
    total_entries: int
    total_exits: int
    alert: AlertStatus


@dataclass
class FrameResult:
    """Everything produced for one processed frame."""

    people: List[TrackedPerson]
    metrics: FrameMetrics
    entered_ids: List[int] = field(default_factory=list)
    exited_ids: List[int] = field(default_factory=list)
    inference_ms: float = 0.0


@dataclass
class _TrackState:
    """Internal per-ID bookkeeping for people currently in the queue."""

    entry_ts: float
    last_inside_ts: float


# ====================================================================== alert engine
class CongestionAlertEngine:
    """Debounced congestion detector (M consecutive frames over a threshold)."""

    def __init__(self, config: AlertConfig) -> None:
        """Create the engine.

        Args:
            config: Alert thresholds.

        Raises:
            ValueError: If thresholds are invalid.
        """
        config.validate()
        self._config = config
        self._queue_streak = 0
        self._wait_streak = 0
        self._status = AlertStatus()

    @property
    def status(self) -> AlertStatus:
        """Most recent alert status."""
        return self._status

    def update_config(self, config: AlertConfig) -> None:
        """Replace thresholds (streak counters are kept)."""
        config.validate()
        self._config = config

    def reset(self) -> None:
        """Clear streak counters and the current status."""
        self._queue_streak = 0
        self._wait_streak = 0
        self._status = AlertStatus()

    def evaluate(self, queue_length: int, wait_seconds: float) -> AlertStatus:
        """Update streaks with one frame's metrics and return the alert status.

        Args:
            queue_length: People currently in the queue.
            wait_seconds: Wait estimate to compare with `max_wait_seconds`.

        Returns:
            Current AlertStatus.
        """
        cfg = self._config
        self._queue_streak = self._queue_streak + 1 if queue_length > cfg.max_queue_length else 0
        self._wait_streak = self._wait_streak + 1 if wait_seconds > cfg.max_wait_seconds else 0
        queue_trig = self._queue_streak >= cfg.consecutive_frames
        wait_trig = self._wait_streak >= cfg.consecutive_frames
        reasons: List[str] = []
        if queue_trig:
            reasons.append(f"queue {queue_length} > {cfg.max_queue_length}")
        if wait_trig:
            reasons.append(f"wait {wait_seconds:.0f}s > {cfg.max_wait_seconds:.0f}s")
        self._status = AlertStatus(
            active=queue_trig or wait_trig,
            queue_triggered=queue_trig,
            wait_triggered=wait_trig,
            reasons=tuple(reasons),
        )
        return self._status


# ====================================================================== analyzer
class QueueAnalyzer:
    """Detects, tracks and times people inside a queue polygon.

    Timestamps are supplied by the caller (video time for files, wall time for streams),
    so results are reproducible and independent of processing speed.
    """

    def __init__(
        self,
        zone: PolygonZone,
        config: Optional[AnalyzerConfig] = None,
        alert_config: Optional[AlertConfig] = None,
    ) -> None:
        """Initialise the analyzer and load the YOLO weights.

        Args:
            zone: Queue polygon.
            config: Detector/tracker settings (defaults to YOLOv8n + ByteTrack).
            alert_config: Alert thresholds.

        Raises:
            ValueError: If a configuration is invalid.
            RuntimeError: If ultralytics is missing or the weights cannot be loaded.
        """
        self._zone = zone
        self._config = config or AnalyzerConfig()
        self._config.validate()
        self._alert_engine = CongestionAlertEngine(alert_config or AlertConfig())
        self._device = self._resolve_device(self._config.device)
        self._half = bool(self._config.half and self._device.startswith("cuda"))
        self._model: Any = None
        self._load_model()
        self._reset_state()

    # ------------------------------------------------------------------ properties
    @property
    def zone(self) -> PolygonZone:
        """The queue polygon."""
        return self._zone

    @property
    def device(self) -> str:
        """Device currently used for inference."""
        return self._device

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Resolve 'auto' to cuda:0 / mps / cpu depending on availability."""
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except ImportError:
            logger.warning("PyTorch not importable; defaulting to CPU.")
        return "cpu"

    def _load_model(self) -> None:
        """Load (or reload) the YOLO model. Reloading also resets the tracker state."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
        try:
            self._model = YOLO(self._config.model_path)
        except Exception as exc:  # noqa: BLE001 - ultralytics raises many exception types
            raise RuntimeError(f"Could not load YOLO weights '{self._config.model_path}': {exc}") from exc
        logger.info("Loaded %s on %s (half=%s).", self._config.model_path, self._device, self._half)

    def _reset_state(self) -> None:
        """Reset all counters and per-track state."""
        self._active: Dict[int, _TrackState] = {}
        self._completed_waits: Deque[float] = deque(maxlen=self._config.wait_window)
        self._event_log: Deque[Dict[str, float]] = deque(maxlen=MAX_EVENT_LOG)
        self._total_entries = 0
        self._total_exits = 0
        self._last_timestamp = 0.0
        self._alert_engine.reset()

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        """Start a fresh analysis (clears metrics and reloads the model to reset tracker IDs)."""
        self._load_model()
        self._reset_state()

    def update_alert_config(self, alert_config: AlertConfig) -> None:
        """Change alert thresholds on the fly."""
        self._alert_engine.update_config(alert_config)

    def get_event_log(self) -> List[Dict[str, float]]:
        """Return completed queue visits: track_id, entry_ts, exit_ts, wait_seconds."""
        return list(self._event_log)

    def save_event_log_csv(self, path: Union[str, Path]) -> Path:
        """Write the completed-visit log to CSV.

        Args:
            path: Destination file.

        Returns:
            The written path.

        Raises:
            OSError: If the file cannot be written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(self._event_log)
        return out

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        """Run detection, tracking and queue analytics on one frame.

        Args:
            frame: BGR image.
            timestamp: Frame time in seconds (monotonic non-decreasing).

        Returns:
            FrameResult. An invalid/empty frame yields an empty result without updating alerts.

        Raises:
            RuntimeError: If inference fails and cannot be recovered (e.g. CPU fallback also fails).
        """
        if not self._is_valid_frame(frame):
            logger.warning("Skipping invalid or empty frame.")
            return self._build_result([], self._last_timestamp, [], [], 0.0, evaluate_alerts=False)
        ts = float(timestamp) if math.isfinite(timestamp) else self._last_timestamp
        ts = max(ts, self._last_timestamp)
        self._last_timestamp = ts

        started = time.perf_counter()
        detections = self._detect(frame)
        inference_ms = (time.perf_counter() - started) * 1000.0

        people, entered = self._update_membership(detections, ts)
        exited = self._expire_tracks(ts)
        return self._build_result(people, ts, entered, exited, inference_ms, evaluate_alerts=True)

    # ------------------------------------------------------------------ detection
    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        """Return True for a non-empty 3-channel uint8-like image."""
        return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        """Return True if the exception is a CUDA out-of-memory error."""
        return "out of memory" in str(exc).lower()

    def _fallback_to_cpu(self) -> None:
        """Free GPU memory and reload the model on CPU (tracker IDs restart)."""
        logger.error("CUDA out of memory - falling back to CPU. Tracker IDs will restart.")
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must never mask the original problem
            pass
        self._device = "cpu"
        self._half = False
        self._load_model()

    def _run_tracker(self, frame: np.ndarray) -> Any:
        """Call ultralytics `track` with one automatic CUDA-OOM recovery."""
        cfg = self._config
        for attempt in range(2):
            try:
                results = self._model.track(
                    frame,
                    persist=True,
                    tracker=cfg.tracker,
                    classes=[cfg.person_class_id],
                    conf=cfg.conf,
                    iou=cfg.iou,
                    imgsz=cfg.imgsz,
                    device=self._device,
                    half=self._half,
                    verbose=False,
                )
                return results[0]
            except RuntimeError as exc:
                if attempt == 0 and self._device != "cpu" and self._is_cuda_oom(exc):
                    self._fallback_to_cpu()
                    continue
                raise
        raise RuntimeError("Tracker inference failed.")  # pragma: no cover

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 + ByteTrack and return detections that have a track ID."""
        result = self._run_tracker(frame)
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        return [
            Detection(tuple(float(v) for v in box), float(conf), int(tid))  # type: ignore[arg-type]
            for box, conf, tid in zip(xyxy, confs, ids)
        ]

    # ------------------------------------------------------------------ queue logic
    def _update_membership(self, detections: List[Detection], timestamp: float) -> Tuple[List[TrackedPerson], List[int]]:
        """Update per-ID ROI state and build the list of people for the current frame."""
        people: List[TrackedPerson] = []
        entered: List[int] = []
        for det in detections:
            feet = bottom_center(det.box)
            inside = self._zone.contains(feet)
            state = self._active.get(det.track_id)
            dwell = 0.0
            if inside:
                if state is None:
                    state = _TrackState(entry_ts=timestamp, last_inside_ts=timestamp)
                    self._active[det.track_id] = state
                    self._total_entries += 1
                    entered.append(det.track_id)
                else:
                    state.last_inside_ts = timestamp
                dwell = timestamp - state.entry_ts
            elif state is not None:
                dwell = timestamp - state.entry_ts  # briefly outside: still within the grace window
            box_int = tuple(int(round(v)) for v in det.box)
            people.append(
                TrackedPerson(
                    track_id=det.track_id,
                    box=box_int,  # type: ignore[arg-type]
                    confidence=det.confidence,
                    feet=feet,
                    in_roi=inside,
                    dwell_seconds=max(0.0, dwell),
                )
            )
        return people, entered

    def _expire_tracks(self, timestamp: float) -> List[int]:
        """Close visits of IDs unseen inside the ROI for longer than the grace period."""
        grace = self._config.exit_grace_seconds
        expired = [tid for tid, st in self._active.items() if timestamp - st.last_inside_ts > grace]
        for tid in expired:
            state = self._active.pop(tid)
            wait = max(0.0, state.last_inside_ts - state.entry_ts)
            self._total_exits += 1
            if wait >= self._config.min_wait_seconds:
                self._completed_waits.append(wait)
                self._event_log.append(
                    {
                        "track_id": float(tid),
                        "entry_ts": state.entry_ts,
                        "exit_ts": state.last_inside_ts,
                        "wait_seconds": wait,
                    }
                )
        return expired

    def _build_result(
        self,
        people: List[TrackedPerson],
        timestamp: float,
        entered: List[int],
        exited: List[int],
        inference_ms: float,
        evaluate_alerts: bool,
    ) -> FrameResult:
        """Compute frame metrics and (optionally) advance the alert engine.

        The wait estimate used for alerting is max(rolling mean of completed waits,
        mean dwell of people currently queued). Taking the maximum reacts early when a
        queue is stuck (nobody has left yet) without ignoring historical waits.
        """
        dwells = [timestamp - st.entry_ts for st in self._active.values()]
        queue_length = len(dwells)
        current_avg = float(np.mean(dwells)) if dwells else 0.0
        max_dwell = float(max(dwells)) if dwells else 0.0
        avg_wait = float(np.mean(self._completed_waits)) if self._completed_waits else 0.0
        effective = max(avg_wait, current_avg)
        alert = self._alert_engine.evaluate(queue_length, effective) if evaluate_alerts else self._alert_engine.status
        metrics = FrameMetrics(
            timestamp=timestamp,
            queue_length=queue_length,
            avg_wait_seconds=avg_wait,
            current_avg_dwell_seconds=current_avg,
            max_dwell_seconds=max_dwell,
            effective_wait_seconds=effective,
            total_entries=self._total_entries,
            total_exits=self._total_exits,
            alert=alert,
        )
        return FrameResult(people=people, metrics=metrics, entered_ids=entered, exited_ids=exited, inference_ms=inference_ms)
