"""Conservatively exclude globally blurred COLMAP training images.

The source dataset is never modified.  A directory containing links to the
accepted images and an auditable CSV/JSON report are created instead.
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-remove-ratio", type=float, default=0.05)
    parser.add_argument("--tail-quantile", type=float, default=0.12)
    parser.add_argument("--neighbor-window", type=int, default=3)
    parser.add_argument("--min-neighbor-ratio", type=float, default=1.45)
    parser.add_argument("--max-side", type=int, default=1280)
    return parser.parse_args()


def robust_gray(path, max_side):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot decode {path}")
    scale = min(1.0, max_side / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return image


def sharpness_metrics(gray):
    height, width = gray.shape
    margin_y, margin_x = int(height * 0.15), int(width * 0.15)
    center = gray[margin_y : height - margin_y, margin_x : width - margin_x]

    def measure(region):
        region = cv2.GaussianBlur(region, (3, 3), 0)
        lap = cv2.Laplacian(region, cv2.CV_32F)
        gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        # The 90th percentile makes the score less dependent on the amount of
        # sky or flat walls than a plain mean-gradient score.
        return float(lap.var()), float(np.percentile(magnitude, 90))

    lap, tenengrad = measure(gray)
    center_lap, center_tenengrad = measure(center)
    return {
        "laplacian": lap,
        "tenengrad_p90": tenengrad,
        "center_laplacian": center_lap,
        "center_tenengrad_p90": center_tenengrad,
    }


def robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    logged = np.log1p(values)
    median = np.median(logged)
    mad = np.median(np.abs(logged - median))
    return (logged - median) / max(1.4826 * mad, 1e-6)


def link_or_copy(source, destination):
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def main():
    args = parse_args()
    if not 0.0 <= args.max_remove_ratio <= 0.10:
        raise ValueError("max-remove-ratio must be between 0 and 0.10")
    if not 0.0 < args.tail_quantile <= 0.20:
        raise ValueError("tail-quantile must be in (0, 0.20]")

    image_dir = args.source / "images"
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) < 10:
        raise RuntimeError(f"Only {len(paths)} images found in {image_dir}")

    rows = []
    for path in paths:
        metrics = sharpness_metrics(robust_gray(path, args.max_side))
        rows.append({"name": path.name, "path": path, **metrics})

    metric_names = [
        "laplacian",
        "tenengrad_p90",
        "center_laplacian",
        "center_tenengrad_p90",
    ]
    for metric in metric_names:
        values = np.array([row[metric] for row in rows])
        z_values = robust_z(values)
        cutoff = float(np.quantile(values, args.tail_quantile))
        for row, z_value in zip(rows, z_values):
            row[f"{metric}_z"] = float(z_value)
            row[f"{metric}_tail"] = row[metric] <= cutoff

    # A geometric mean of the four normalized scores is only used for ranking.
    # Removal still requires every independent condition below.
    scores = np.mean(
        [[row[f"{metric}_z"] for metric in metric_names] for row in rows],
        axis=1,
    )
    for row, score in zip(rows, scores):
        row["score"] = float(score)

    for index, row in enumerate(rows):
        lo = max(0, index - args.neighbor_window)
        hi = min(len(rows), index + args.neighbor_window + 1)
        left_scores = [rows[j]["score"] for j in range(lo, index)]
        right_scores = [rows[j]["score"] for j in range(index + 1, hi)]
        neighbor_scores = left_scores + right_scores
        neighbor_median = float(np.median(neighbor_scores))
        # Convert the log-domain score gap back to an intuitive ratio.
        row["neighbor_score_gap"] = neighbor_median - row["score"]
        row["neighbor_outlier"] = (
            row["neighbor_score_gap"] >= np.log(args.min_neighbor_ratio)
        )
        required_gap = np.log(args.min_neighbor_ratio)
        # A removable frame must be bracketed by sharper observations on both
        # sides of the capture sequence. This is a conservative proxy for pose
        # redundancy and prevents dropping an isolated but informative view.
        row["bracketed_by_sharp_frames"] = (
            bool(left_scores)
            and bool(right_scores)
            and max(left_scores) - row["score"] >= required_gap
            and max(right_scores) - row["score"] >= required_gap
        )
        row["all_metrics_tail"] = all(
            row[f"{metric}_tail"] for metric in metric_names
        )
        row["strong_global_outlier"] = max(
            row[f"{metric}_z"] for metric in metric_names
        ) <= -1.25
        row["candidate"] = (
            row["all_metrics_tail"]
            and row["strong_global_outlier"]
            and row["neighbor_outlier"]
            and row["bracketed_by_sharp_frames"]
        )

    # Keep the two ends of a capture and never remove adjacent frames. This
    # protects trajectory coverage without trusting filename gaps as geometry.
    candidates = sorted(
        (i for i, row in enumerate(rows) if row["candidate"]),
        key=lambda i: rows[i]["score"],
    )
    limit = int(np.floor(len(rows) * args.max_remove_ratio))
    removed = set()
    for index in candidates:
        if len(removed) >= limit:
            break
        if index < 2 or index >= len(rows) - 2:
            continue
        if any(abs(index - chosen) <= 1 for chosen in removed):
            continue
        removed.add(index)

    args.output.mkdir(parents=True, exist_ok=True)
    selected_dir = args.output / "images"
    selected_dir.mkdir(exist_ok=True)
    for old in selected_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()

    for index, row in enumerate(rows):
        row["decision"] = "remove_blur" if index in removed else "keep"
        if index not in removed:
            link_or_copy(row["path"], selected_dir / row["name"])

    report_fields = [
        "name",
        *metric_names,
        "score",
        "neighbor_score_gap",
        "all_metrics_tail",
        "strong_global_outlier",
        "neighbor_outlier",
        "bracketed_by_sharp_frames",
        "candidate",
        "decision",
    ]
    with (args.output / "blur_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in report_fields} for row in rows
        )

    summary = {
        "source": str(args.source),
        "selected_images": str(selected_dir),
        "total": len(rows),
        "kept": len(rows) - len(removed),
        "removed": len(removed),
        "removed_ratio": len(removed) / len(rows),
        "removed_names": [rows[i]["name"] for i in sorted(removed)],
        "policy": {
            "max_remove_ratio": args.max_remove_ratio,
            "tail_quantile": args.tail_quantile,
            "neighbor_window": args.neighbor_window,
            "min_neighbor_ratio": args.min_neighbor_ratio,
            "requires_all_four_metrics": True,
            "requires_sharper_frames_on_both_sides": True,
            "protect_capture_ends": True,
            "prevent_adjacent_removals": True,
        },
    }
    (args.output / "blur_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
