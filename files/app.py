"""Streamlit dashboard for the Intelligent Queue & Waiting Time Analyzer.

Run with:  streamlit run app.py

Design note: the analysis stores per-frame raw metrics once. Alert thresholds are then
re-evaluated ("replayed") on those metrics, so moving the sliders updates charts and alert
counts instantly without re-running YOLO. A new run applies the thresholds to the live overlay.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from geometry import PolygonError, PolygonZone, RoiConfig, parse_roi_dict
from overlay import draw_overlay, draw_zone
from tracker_engine import AlertConfig, AnalyzerConfig, CongestionAlertEngine, QueueAnalyzer
from video_io import FrameSource, VideoSink, grab_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("queue_analyzer.app")

LIVE_CHART_EVERY = 10  # processed frames between live chart refreshes
LIVE_IMAGE_EVERY = 2  # processed frames between live image refreshes


@dataclass(frozen=True)
class RunSettings:
    """Sidebar settings for one analysis run."""

    model_path: str
    tracker: str
    conf: float
    imgsz: int
    frame_stride: int
    max_frames: int


# ====================================================================== helpers
def persist_upload(uploaded: Any) -> Path:
    """Write an uploaded video to a temp file once per (name, size) and return its path."""
    key = (uploaded.name, uploaded.size)
    cached = st.session_state.get("video_key")
    path = st.session_state.get("video_path")
    if cached == key and path and Path(path).is_file():
        return Path(path)
    if path and Path(path).is_file():
        Path(path).unlink(missing_ok=True)
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getbuffer())
        new_path = Path(handle.name)
    st.session_state["video_key"] = key
    st.session_state["video_path"] = str(new_path)
    st.session_state.pop("analysis", None)  # results belong to the previous video
    return new_path


def load_roi_from_upload(uploaded: Any) -> Optional[RoiConfig]:
    """Parse an uploaded roi_config.json, showing a friendly error on failure."""
    if uploaded is None:
        return None
    try:
        return parse_roi_dict(json.loads(uploaded.getvalue().decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError, PolygonError) as exc:
        st.sidebar.error(f"Invalid ROI file: {exc}")
        return None


def default_polygon(width: int, height: int) -> np.ndarray:
    """Centered rectangle covering the middle 60% of the frame (fallback ROI)."""
    x0, x1 = int(width * 0.2), int(width * 0.8)
    y0, y1 = int(height * 0.2), int(height * 0.8)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def replay_alerts(frames: pd.DataFrame, alert_cfg: AlertConfig) -> pd.DataFrame:
    """Re-evaluate alert logic on stored per-frame metrics with new thresholds.

    Args:
        frames: DataFrame with `queue_length` and `effective_wait_seconds` columns.
        alert_cfg: Thresholds to apply.

    Returns:
        Copy with `alert_active`, `queue_alert` and `wait_alert` boolean columns.
    """
    engine = CongestionAlertEngine(alert_cfg)
    active: List[bool] = []
    queue_flags: List[bool] = []
    wait_flags: List[bool] = []
    for queue_len, wait in zip(frames["queue_length"], frames["effective_wait_seconds"]):
        status = engine.evaluate(int(queue_len), float(wait))
        active.append(status.active)
        queue_flags.append(status.queue_triggered)
        wait_flags.append(status.wait_triggered)
    out = frames.copy()
    out["alert_active"] = active
    out["queue_alert"] = queue_flags
    out["wait_alert"] = wait_flags
    return out


def alert_segments(frames: pd.DataFrame) -> pd.DataFrame:
    """Summarise contiguous alert intervals (start, end, duration in seconds)."""
    rows: List[Dict[str, float]] = []
    start: Optional[float] = None
    last_ts = 0.0
    for ts, flag in zip(frames["timestamp"], frames["alert_active"]):
        if flag and start is None:
            start = float(ts)
        if not flag and start is not None:
            rows.append({"start_s": start, "end_s": last_ts, "duration_s": last_ts - start})
            start = None
        last_ts = float(ts)
    if start is not None:
        rows.append({"start_s": start, "end_s": last_ts, "duration_s": last_ts - start})
    return pd.DataFrame(rows, columns=["start_s", "end_s", "duration_s"])


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for st.image."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ====================================================================== analysis
def run_analysis(
    video_path: Path,
    roi: Optional[RoiConfig],
    settings: RunSettings,
    alert_cfg: AlertConfig,
) -> Optional[Dict[str, Any]]:
    """Process the whole video, streaming annotated frames and charts into the page.

    Args:
        video_path: Local video file.
        roi: ROI configuration, or None to use the default rectangle.
        settings: Detector/tracker/run settings.
        alert_cfg: Thresholds applied to the live overlay.

    Returns:
        Dict with the stored analysis, or None if it failed.
    """
    image_ph = st.empty()
    progress = st.progress(0.0, text="Starting...")
    cols = st.columns(4)
    metric_phs = [c.empty() for c in cols]
    chart_ph = st.empty()
    rows: List[Dict[str, float]] = []
    out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
    sink: Optional[VideoSink] = None
    started = time.perf_counter()
    try:
        with FrameSource(str(video_path)) as source:
            if source.width <= 0 or source.height <= 0:
                st.error("The video did not report a valid frame size.")
                return None
            polygon = roi.scaled_polygon(source.width, source.height) if roi else default_polygon(source.width, source.height)
            zone = PolygonZone(polygon)
            analyzer = QueueAnalyzer(
                zone,
                AnalyzerConfig(
                    model_path=settings.model_path,
                    tracker=settings.tracker,
                    conf=settings.conf,
                    imgsz=settings.imgsz,
                ),
                alert_cfg,
            )
            sink = VideoSink(out_path, source.fps / settings.frame_stride, (source.width, source.height))
            total = source.frame_count
            frame_index = -1
            processed = 0
            while True:
                item = source.read()
                if item is None:
                    break
                frame, timestamp = item
                frame_index += 1
                if frame_index % settings.frame_stride:
                    continue
                result = analyzer.process_frame(frame, timestamp)
                processed += 1
                m = result.metrics
                annotated = draw_overlay(frame, result, zone, fps=processed / max(time.perf_counter() - started, 1e-6))
                sink.write(annotated)
                rows.append(
                    {
                        "timestamp": m.timestamp,
                        "queue_length": m.queue_length,
                        "avg_wait_seconds": m.avg_wait_seconds,
                        "current_avg_dwell_seconds": m.current_avg_dwell_seconds,
                        "effective_wait_seconds": m.effective_wait_seconds,
                        "total_entries": m.total_entries,
                        "total_exits": m.total_exits,
                        "inference_ms": result.inference_ms,
                    }
                )
                if processed % LIVE_IMAGE_EVERY == 0:
                    image_ph.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
                if processed % LIVE_CHART_EVERY == 0:
                    metric_phs[0].metric("Queue length", m.queue_length)
                    metric_phs[1].metric("Avg wait (s)", f"{m.avg_wait_seconds:.0f}")
                    metric_phs[2].metric("Longest wait (s)", f"{m.max_dwell_seconds:.0f}")
                    metric_phs[3].metric("Alert", "ACTIVE" if m.alert.active else "OK")
                    chart_ph.line_chart(pd.DataFrame(rows).set_index("timestamp")[["queue_length"]])
                    if total > 0:
                        progress.progress(min(1.0, (frame_index + 1) / total), text=f"Frame {frame_index + 1}/{total}")
                if settings.max_frames and processed >= settings.max_frames:
                    break
            events = pd.DataFrame(analyzer.get_event_log(), columns=["track_id", "entry_ts", "exit_ts", "wait_seconds"])
            fps_out = source.fps / settings.frame_stride
    except (FileNotFoundError, RuntimeError, ValueError, PolygonError) as exc:
        st.error(f"Analysis failed: {exc}")
        return None
    finally:
        if sink is not None:
            sink.release()

    progress.progress(1.0, text="Done")
    if not rows:
        st.warning("No frames were processed.")
        return None
    video_bytes = out_path.read_bytes() if out_path.is_file() else b""
    out_path.unlink(missing_ok=True)
    return {
        "frames": pd.DataFrame(rows),
        "events": events,
        "video_bytes": video_bytes,
        "fps": fps_out,
        "wall_seconds": time.perf_counter() - started,
    }


# ====================================================================== rendering
def render_results(analysis: Dict[str, Any], alert_cfg: AlertConfig) -> None:
    """Render summary metrics, threshold-aware charts, alert intervals and downloads."""
    frames = replay_alerts(analysis["frames"], alert_cfg)
    events: pd.DataFrame = analysis["events"]
    processed = len(frames)
    wall = analysis["wall_seconds"]

    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Peak queue length", int(frames["queue_length"].max()))
    c2.metric("Mean queue length", f"{frames['queue_length'].mean():.1f}")
    c3.metric("Mean completed wait (s)", f"{events['wait_seconds'].mean():.0f}" if len(events) else "n/a")
    c4.metric("Alert frames", int(frames["alert_active"].sum()))
    c5.metric("Processing FPS", f"{processed / max(wall, 1e-6):.1f}")

    st.subheader("Queue length over time")
    chart_q = frames.set_index("timestamp")[["queue_length"]].copy()
    chart_q["max queue threshold"] = alert_cfg.max_queue_length
    st.line_chart(chart_q)

    st.subheader("Waiting time over time (seconds)")
    chart_w = frames.set_index("timestamp")[["avg_wait_seconds", "current_avg_dwell_seconds"]].copy()
    chart_w["max wait threshold"] = alert_cfg.max_wait_seconds
    st.line_chart(chart_w)

    st.subheader("Congestion alerts (with current sliders)")
    segments = alert_segments(frames)
    if segments.empty:
        st.success("No alert interval with the current thresholds.")
    else:
        st.warning(f"{len(segments)} alert interval(s) detected.")
        st.dataframe(segments.round(1), use_container_width=True)

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    if analysis["video_bytes"]:
        d1.download_button("Annotated video (.mp4)", analysis["video_bytes"], "annotated.mp4", "video/mp4")
    d2.download_button("Per-frame metrics (.csv)", frames.to_csv(index=False).encode("utf-8"), "metrics.csv", "text/csv")
    d3.download_button("Completed visits (.csv)", events.to_csv(index=False).encode("utf-8"), "visits.csv", "text/csv")
    st.caption("Annotated .mp4 uses the mp4v codec; some browsers cannot play it inline, so it is offered as a download.")


# ====================================================================== page
def sidebar_controls() -> Tuple[AlertConfig, RunSettings, Any, bool]:
    """Draw the sidebar and return (alert config, run settings, ROI upload, run clicked)."""
    st.sidebar.header("Alert thresholds")
    max_queue = st.sidebar.slider("Max queue length (people)", 1, 50, 8)
    max_wait = st.sidebar.slider("Max average wait time (seconds)", 5, 1800, 120, step=5)
    consecutive = st.sidebar.slider("Consecutive frames (M)", 1, 120, 15)
    st.sidebar.header("Detector & tracker")
    model = st.sidebar.selectbox("YOLOv8 model", ["yolov8n.pt", "yolov8s.pt"])
    tracker = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, step=0.05)
    imgsz = st.sidebar.select_slider("Inference size", options=[480, 640, 800, 960], value=640)
    stride = st.sidebar.slider("Process every k-th frame", 1, 10, 2)
    max_frames = st.sidebar.number_input("Max processed frames (0 = all)", min_value=0, value=0, step=100)
    st.sidebar.header("Queue zone")
    roi_upload = st.sidebar.file_uploader("roi_config.json (from roi_selector.py)", type=["json"])
    run_clicked = st.sidebar.button("Run analysis", type="primary", use_container_width=True)
    st.sidebar.caption("Changing a slider during a run restarts the page; stored results are kept.")
    alert_cfg = AlertConfig(max_queue_length=max_queue, max_wait_seconds=float(max_wait), consecutive_frames=consecutive)
    settings = RunSettings(model, tracker, conf, int(imgsz), int(stride), int(max_frames))
    return alert_cfg, settings, roi_upload, bool(run_clicked)


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Queue & Waiting Time Analyzer", layout="wide")
    st.title("Intelligent Queue & Waiting Time Analyzer")
    st.caption("Smart campus services: cafeteria, registration desk, library. Anonymous tracking only - no faces, no identities.")
    alert_cfg, settings, roi_upload, run_clicked = sidebar_controls()

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.info("Upload a video to begin. Create the queue zone beforehand with `python roi_selector.py`.")
        return
    video_path = persist_upload(uploaded)
    roi = load_roi_from_upload(roi_upload)

    try:
        preview = grab_frame(str(video_path), 0)
        height, width = preview.shape[:2]
        polygon = roi.scaled_polygon(width, height) if roi else default_polygon(width, height)
        st.image(bgr_to_rgb(draw_zone(preview, PolygonZone(polygon))), caption="Queue zone preview", use_container_width=True)
    except (RuntimeError, PolygonError) as exc:
        st.error(f"Cannot preview the video: {exc}")
        return
    if roi is None:
        st.warning("No ROI file uploaded: using a default centered rectangle. Upload roi_config.json for accurate results.")

    if run_clicked:
        analysis = run_analysis(video_path, roi, settings, alert_cfg)
        if analysis is not None:
            st.session_state["analysis"] = analysis
            st.rerun()
    analysis_state = st.session_state.get("analysis")
    if analysis_state is not None:
        render_results(analysis_state, alert_cfg)


if __name__ == "__main__":
    main()
