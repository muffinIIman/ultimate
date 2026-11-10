"""Offline evaluation against manual ground truth.

Inputs:
    --metrics   per-frame CSV written by `main.py --metrics-csv`
    --events    completed-visits CSV written by `main.py --event-log`
    --gt-counts CSV with columns: timestamp,count      (manual queue counts, e.g. every 5 s)
    --gt-events CSV with columns: entry_ts,exit_ts     (manual entry/exit times per person)

Example:
    python evaluate.py --metrics outputs/metrics.csv --gt-counts gt/counts.csv \
        --events outputs/events.csv --gt-events gt/events.csv --tolerance 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("evaluate")


def queue_count_errors(metrics: pd.DataFrame, gt_counts: pd.DataFrame) -> dict:
    """Compare predicted queue length with manual counts (nearest predicted frame in time).

    Args:
        metrics: DataFrame with `timestamp` and `queue_length`.
        gt_counts: DataFrame with `timestamp` and `count`.

    Returns:
        Dict with MAE, RMSE, bias and sample size.

    Raises:
        ValueError: If required columns are missing or the inputs are empty.
    """
    for frame, cols in ((metrics, ("timestamp", "queue_length")), (gt_counts, ("timestamp", "count"))):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    if metrics.empty or gt_counts.empty:
        raise ValueError("Metrics and ground-truth counts must not be empty.")
    pred = metrics.sort_values("timestamp")
    gt = gt_counts.sort_values("timestamp")
    merged = pd.merge_asof(gt, pred[["timestamp", "queue_length"]], on="timestamp", direction="nearest")
    err = merged["queue_length"].to_numpy(dtype=float) - merged["count"].to_numpy(dtype=float)
    return {
        "n_samples": int(len(err)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def match_times(pred: np.ndarray, gt: np.ndarray, tolerance: float) -> Tuple[int, int, int, List[Tuple[int, int]]]:
    """Greedy one-to-one matching of predicted and ground-truth event times.

    Args:
        pred: Predicted event times (seconds).
        gt: Ground-truth event times (seconds).
        tolerance: Maximum absolute time difference for a match.

    Returns:
        (true positives, false positives, false negatives, matched (pred_idx, gt_idx) pairs).
    """
    used: set = set()
    pairs: List[Tuple[int, int]] = []
    for gi, g in enumerate(gt):
        best: Optional[int] = None
        best_diff = tolerance + 1e-9
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            diff = abs(p - g)
            if diff <= best_diff:
                best, best_diff = pi, diff
        if best is not None:
            used.add(best)
            pairs.append((best, gi))
    tp = len(pairs)
    return tp, len(pred) - tp, len(gt) - tp, pairs


def prf(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall and F1 from counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1}


def event_metrics(events: pd.DataFrame, gt_events: pd.DataFrame, tolerance: float) -> dict:
    """Precision/recall for entry and exit events plus wait-time MAE on matched visits.

    Args:
        events: Predicted visits with `entry_ts`, `exit_ts`.
        gt_events: Ground-truth visits with `entry_ts`, `exit_ts`.
        tolerance: Matching tolerance in seconds.

    Returns:
        Nested dict of metrics.

    Raises:
        ValueError: If required columns are missing.
    """
    for frame in (events, gt_events):
        missing = [c for c in ("entry_ts", "exit_ts") if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    pe, ge = events["entry_ts"].to_numpy(float), gt_events["entry_ts"].to_numpy(float)
    px, gx = events["exit_ts"].to_numpy(float), gt_events["exit_ts"].to_numpy(float)
    tp_e, fp_e, fn_e, pairs = match_times(pe, ge, tolerance)
    tp_x, fp_x, fn_x, _ = match_times(px, gx, tolerance)
    wait_err = [abs((px[p] - pe[p]) - (gx[g] - ge[g])) for p, g in pairs]
    return {
        "entry": prf(tp_e, fp_e, fn_e),
        "exit": prf(tp_x, fp_x, fn_x),
        "wait_time_MAE_seconds": float(np.mean(wait_err)) if wait_err else None,
        "matched_visits": len(pairs),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns an exit code."""
    parser = argparse.ArgumentParser(description="Evaluate queue analytics against manual ground truth.")
    parser.add_argument("--metrics", help="Per-frame metrics CSV.")
    parser.add_argument("--gt-counts", help="Ground-truth counts CSV (timestamp,count).")
    parser.add_argument("--events", help="Predicted visits CSV.")
    parser.add_argument("--gt-events", help="Ground-truth visits CSV (entry_ts,exit_ts).")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Event matching tolerance in seconds.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.metrics and args.gt_counts:
            stats = queue_count_errors(pd.read_csv(args.metrics), pd.read_csv(args.gt_counts))
            logger.info("Queue count: %s", stats)
        if args.events and args.gt_events:
            stats = event_metrics(pd.read_csv(args.events), pd.read_csv(args.gt_events), args.tolerance)
            logger.info("Entry: %s", stats["entry"])
            logger.info("Exit: %s", stats["exit"])
            logger.info("Wait-time MAE (s): %s over %d matched visits", stats["wait_time_MAE_seconds"], stats["matched_visits"])
        if not ((args.metrics and args.gt_counts) or (args.events and args.gt_events)):
            parser.error("Provide --metrics with --gt-counts and/or --events with --gt-events.")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
