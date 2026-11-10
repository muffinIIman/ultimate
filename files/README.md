# Intelligent Queue & Waiting Time Analyzer

Measures queue length and waiting time for campus services (cafeteria, registration desk, library)
from a video file or RTSP stream, using YOLOv8 + ByteTrack + OpenCV, with a Streamlit dashboard.

## 1. Architecture and data flow

```
Frame ingestion        video_io.FrameSource   (file / RTSP / webcam, reconnect, video-time timestamps)
      |
Object detection       tracker_engine.QueueAnalyzer._run_tracker   (YOLOv8n/s, class = person)
      |
Multi-object tracking  ultralytics ByteTrack (persist=True)        -> stable track IDs
      |
Polygon filtering      geometry.PolygonZone   feet = ((x1+x2)/2, y2) -> point-in-polygon (Shapely / cv2)
      |
Queue & wait metrics   QueueAnalyzer: entry/exit per ID, grace period, rolling mean of completed waits
      |
Alert engine           CongestionAlertEngine: (queue > N) OR (wait > T) for M consecutive frames
      |
Presentation           overlay.draw_overlay -> main.py (OpenCV window + mp4)  |  app.py (Streamlit)
```

| File | Responsibility |
|---|---|
| `geometry.py` | polygon validation, `PolygonZone`, ROI JSON load/save with frame-size rescaling |
| `video_io.py` | `FrameSource` (robust reading), `VideoSink`, `grab_frame` |
| `roi_selector.py` | interactive ROI tool, exports `roi_config.json` |
| `tracker_engine.py` | `QueueAnalyzer`, `CongestionAlertEngine`, configs and result dataclasses |
| `overlay.py` | OpenCV drawing shared by CLI and dashboard |
| `main.py` | CLI loop (file or RTSP), annotated video, CSV exports |
| `app.py` | Streamlit dashboard |
| `evaluate.py` | MAE and precision/recall against manual ground truth |

## 2. Quick start

```bash
pip install -r requirements.txt
python roi_selector.py --source data/cafeteria.mp4 --output roi_config.json
python main.py --source data/cafeteria.mp4 --roi roi_config.json \
    --metrics-csv outputs/metrics.csv --event-log outputs/events.csv
streamlit run app.py
```

YOLO weights (`yolov8n.pt`) are downloaded automatically on first use (network required once).
SORT is not bundled by Ultralytics; ByteTrack (default) and BoT-SORT are available, and any custom tracker YAML
can be passed with `--tracker`.

## 3. Wait-time definition

- A person **enters** when the feet point of a tracked ID is first inside the polygon.
- A person **exits** when the ID has not been seen inside the polygon for `exit_grace_seconds` (default 1.5 s).
  The grace period absorbs detection flicker and short occlusions.
- **Completed wait** = last time inside - entry time (visits shorter than `min_wait_seconds` are ignored).
- `avg_wait_seconds` = rolling mean of the last 50 completed waits.
- The alert uses `max(avg_wait_seconds, mean dwell of people currently queued)`, so a stuck queue is detected
  even before anyone has left.

## 4. Evaluation plan

Record 3 to 5 clips per scene (empty, normal, rush hour; different lighting). Annotate manually:
queue count every 5 s (`gt/counts.csv`: `timestamp,count`) and entry/exit times of each person
(`gt/events.csv`: `entry_ts,exit_ts`). Then run `evaluate.py`.

| Metric | Definition | Target |
|---|---|---|
| Queue count MAE | mean absolute difference between predicted and manual queue length at each annotated time | <= 1 person |
| Queue count RMSE and bias | penalises large errors, reveals systematic under/over-counting | bias near 0 |
| Entry precision / recall | predicted entries matched one-to-one to manual entries within +/- 2 s (TP/FP/FN) | >= 0.90 / 0.90 |
| Exit precision / recall | same for exit events (ID switches show up as FP + FN) | >= 0.85 / 0.85 |
| Wait-time MAE | abs difference between predicted and manual wait of matched visits | <= 5 s |
| FPS benchmark | end-to-end FPS and mean inference ms on a fixed 1080p clip | >= 15 FPS on the demo hardware |

FPS benchmark grid: `yolov8n` vs `yolov8s` x `imgsz` 640 vs 960 x CPU vs GPU, with `--frame-stride` 1 and 2.
Report the mean and standard deviation over 3 runs (`main.py` prints end-to-end FPS and mean inference ms).
Run ablations on `conf` (0.25/0.35/0.5) and `exit_grace_seconds` (0.5/1.5/3.0) to justify the defaults.

## 5. Three architectural decisions to defend

1. **Tracking-by-detection with ByteTrack, no appearance model (no ReID).**
   Pro: real-time on modest hardware, no per-person embeddings, no biometric data (privacy-friendly).
   Trade-off: more ID switches in dense, occluded queues. Mitigations: grace period before closing a visit,
   `min_wait_seconds` filtering, and reporting exit precision/recall to quantify the residual error.
2. **Feet point + polygon ROI instead of box center or a counting line.**
   Pro: the feet point lies on the ground plane, so membership is stable under perspective and partial occlusion
   of the upper body; a polygon also supports curved or L-shaped queues, unlike a single line.
   Trade-off: depends on a correct, camera-specific ROI (hence `roi_selector.py` and a stored frame size),
   and feet are cut off when a person is at the frame edge.
3. **Per-ID dwell-time estimator with injected timestamps, instead of a throughput model.**
   Pro: gives a direct observation of each person's wait; video-time timestamps make experiments reproducible
   and independent of processing speed. Trade-off: completed waits lag reality and ignore people still waiting
   (right-censoring), which is why the alert uses `max(completed mean, current mean)`. Little's law
   (W = L / lambda) is the alternative to discuss as future work.

Also worth mentioning: alerts only assist staff (human in the loop), no faces or identities are processed,
and only anonymous IDs and aggregate metrics are stored, in line with the data-protection requirements
of the course's collective project.
