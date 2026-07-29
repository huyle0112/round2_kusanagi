"""Successive-halving validation for paper-faithful GaussianPro on HCM0421."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--final-iterations", type=int, default=30000)
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


TRIALS = [
    {
        "name": "official_like",
        "downsample": 2,
        "patch_radius": 2,
        "photo_error": 0.25,
        "depth_start": 1.0,
        "depth_end": 0.8,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "depth_080_060",
        "downsample": 2,
        "patch_radius": 2,
        "photo_error": 0.25,
        "depth_start": 0.8,
        "depth_end": 0.6,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "depth_060_040",
        "downsample": 2,
        "patch_radius": 2,
        "photo_error": 0.25,
        "depth_start": 0.6,
        "depth_end": 0.4,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "strict_photo_020",
        "downsample": 2,
        "patch_radius": 2,
        "photo_error": 0.20,
        "depth_start": 0.8,
        "depth_end": 0.6,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "strict_photo_015",
        "downsample": 2,
        "patch_radius": 2,
        "photo_error": 0.15,
        "depth_start": 0.8,
        "depth_end": 0.6,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "larger_patch",
        "downsample": 2,
        "patch_radius": 3,
        "photo_error": 0.20,
        "depth_start": 0.8,
        "depth_end": 0.6,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "coarse_geometry",
        "downsample": 4,
        "patch_radius": 2,
        "photo_error": 0.20,
        "depth_start": 0.8,
        "depth_end": 0.6,
        "anchors_step": 128,
        "voxel_factor": 1.0,
        "anchor_multiplier": 1.25,
    },
    {
        "name": "denser_gp",
        "downsample": 2,
        "patch_radius": 2,
        "photo_error": 0.20,
        "depth_start": 0.8,
        "depth_end": 0.6,
        "anchors_step": 192,
        "voxel_factor": 0.75,
        "anchor_multiplier": 1.40,
    },
]


def schedule(iterations):
    if iterations <= 6000:
        return 500, 3000
    if iterations <= 15000:
        return 1000, 6000
    return 1000, 12000


def final_checkpoint(model_dir, iterations):
    return (
        model_dir
        / "point_cloud"
        / f"iteration_{iterations}"
        / "point_cloud.ply"
    )


def read_final_validation(model_dir, iterations):
    metrics = pd.read_csv(model_dir / "validation_metrics.csv")
    rows = metrics[
        (metrics["split"] == "val")
        & (metrics["iteration"] == iterations)
    ]
    if rows.empty:
        raise RuntimeError(
            f"No held-out validation row at {iterations}: {model_dir}"
        )
    row = rows.iloc[-1]
    # Transparent and deterministic ranking; PSNR remains the primary metric.
    score = (
        0.8 * float(row.psnr)
        + 0.2 * float(row.top20_psnr)
        + 10.0 * float(row.ssim)
        - 5.0 * float(row.lpips)
        - 2.0 * float(row.edge_l1)
    )
    return {
        "psnr": float(row.psnr),
        "top20_psnr": float(row.top20_psnr),
        "edge_l1": float(row.edge_l1),
        "ssim": float(row.ssim),
        "lpips": float(row.lpips),
        "score": score,
        "anchors": int(row.anchors),
    }


def run_trial(args, stage_name, iterations, config):
    model_dir = args.output / stage_name / config["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = final_checkpoint(model_dir, iterations)
    metrics_path = model_dir / "validation_metrics.csv"
    if checkpoint.exists() and metrics_path.exists():
        print(f"[RESUME] {stage_name}/{config['name']}")
        return read_final_validation(model_dir, iterations)

    if any(model_dir.iterdir()):
        shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)

    start_iter, propagation_end = schedule(iterations)
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(args.source),
        "-m",
        str(model_dir),
        "--images",
        str(args.images),
        "-r",
        "1",
        "--data_device",
        "cpu",
        "--appearance_dim",
        "0",
        "--gpu",
        args.gpu,
        "--iterations",
        str(iterations),
        "--validation_ratio",
        "0.10",
        "--validation_seed",
        "42",
        "--validation_split_mode",
        "stratified",
        "--validation_sample_count",
        "0",
        "--test_iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--lambda_dssim",
        "0.2",
        "--lambda_edge_init",
        "0.0",
        "--lambda_edge_final",
        "0.0",
        "--correct_radial_distortion",
        "--use_gaussianpro",
        "--gaussianpro_paper_faithful",
        "--gaussianpro_start_iter",
        str(start_iter),
        "--gaussianpro_add_until_iter",
        str(propagation_end),
        "--gaussianpro_refine_until_iter",
        str(propagation_end),
        "--gaussianpro_interval",
        "30",
        "--gaussianpro_neighbors",
        "4",
        "--gaussianpro_downsample",
        str(config["downsample"]),
        "--gaussianpro_patch_radius",
        str(config["patch_radius"]),
        "--gaussianpro_patchmatch_iterations",
        "3",
        "--gaussianpro_min_consistent_views",
        "2",
        "--gaussianpro_relaxed_min_views",
        "2",
        "--gaussianpro_max_photo_error",
        str(config["photo_error"]),
        "--gaussianpro_reprojection_threshold",
        "2.0",
        "--gaussianpro_depth_consistency_threshold",
        "0.01",
        "--gaussianpro_normal_consistency_threshold",
        "0.0",
        "--gaussianpro_depth_discrepancy_start",
        str(config["depth_start"]),
        "--gaussianpro_depth_discrepancy_end",
        str(config["depth_end"]),
        "--gaussianpro_edge_residual_priority",
        "0.0",
        "--gaussianpro_max_anchors_per_step",
        str(config["anchors_step"]),
        "--gaussianpro_voxel_factor",
        str(config["voxel_factor"]),
        "--gaussianpro_max_anchor_multiplier",
        str(config["anchor_multiplier"]),
        "--gaussianpro_scaffold_fallback_interval",
        "200",
        "--lambda_gaussianpro_flatness",
        "0.001",
        "--lambda_gaussianpro_normal_l1",
        "0.001",
        "--lambda_gaussianpro_normal_cos",
        "0.001",
    ]
    (model_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"\n[{stage_name}] {config['name']} ({iterations} iterations)")
    with (model_dir / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
        )
    return read_final_validation(model_dir, iterations)


def run_stage(args, stage_name, iterations, configs):
    records = []
    for config in configs:
        metrics = run_trial(args, stage_name, iterations, config)
        records.append(
            {
                "stage": stage_name,
                "iterations": iterations,
                **config,
                **metrics,
            }
        )
        leaderboard = pd.DataFrame(records).sort_values(
            "score", ascending=False
        )
        leaderboard.to_csv(
            args.output / f"{stage_name}_leaderboard.csv", index=False
        )
        print(leaderboard[[
            "name", "psnr", "top20_psnr", "edge_l1",
            "ssim", "lpips", "score", "anchors"
        ]])
    return sorted(records, key=lambda row: row["score"], reverse=True)


def main():
    args = parse_args()
    args.source = args.source.resolve()
    args.images = args.images.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "search_space.json").write_text(
        json.dumps(TRIALS, indent=2), encoding="utf-8"
    )

    stage1 = run_stage(args, "stage1_6k", 6000, TRIALS)
    top3_names = {row["name"] for row in stage1[:3]}
    top3 = [config for config in TRIALS if config["name"] in top3_names]

    stage2 = run_stage(args, "stage2_15k", 15000, top3)
    best_name = stage2[0]["name"]
    best = next(config for config in TRIALS if config["name"] == best_name)
    summary = {
        "validation_split": "90:10",
        "validation_seed": 42,
        "stage1_top3": [row["name"] for row in stage1[:3]],
        "stage2_best": best,
        "stage2_metrics": stage2[0],
    }

    if not args.skip_final:
        final = run_stage(
            args,
            "stage3_30k",
            args.final_iterations,
            [best],
        )[0]
        summary["final_validation"] = final

    (args.output / "best_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nBEST CONFIG")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
