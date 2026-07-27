"""Kaggle GPU smoke run for full-form GaussianPro on the bonsai scene.

Inputs are supplied as two private Kaggle datasets:

* ``kusanagi-pose``: the local ``pose`` directory, possibly stored as
  per-directory ZIP files by Kaggle CLI.
* ``kusanagi-source-fullform``: ``kusanagi-source.zip`` containing this repo's
  runtime source and CUDA extensions.

The only durable outputs are a CSV-validated submission ZIP and its manifest.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from PIL import Image


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
CODE_DIR = WORK_ROOT / "kusanagi"
EXTRACTED_DATA_DIR = WORK_ROOT / "pose_extracted"
MODEL_DIR = WORK_ROOT / "bonsai_7000_model"
PACKAGE_DIR = WORK_ROOT / "submission_package"
OUTPUT_ZIP = WORK_ROOT / "bonsai_7000_submission.zip"
MANIFEST_PATH = WORK_ROOT / "bonsai_7000_manifest.json"
ITERATIONS = 7_000


def run(command: list[object], label: str, cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print("\n" + "=" * 100)
    print(label)
    print(printable)
    print("=" * 100, flush=True)
    started = time.time()
    subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
    )
    print(f"{label} finished in {(time.time() - started) / 60:.1f} min")


def locate_or_extract_source() -> Path:
    direct_candidates = [
        path.parent
        for path in INPUT_ROOT.rglob("train.py")
        if (path.parent / "gaussian_renderer").is_dir()
    ]
    if direct_candidates:
        if CODE_DIR.exists():
            shutil.rmtree(CODE_DIR)
        shutil.copytree(direct_candidates[0], CODE_DIR)
        return CODE_DIR

    archives = list(INPUT_ROOT.rglob("kusanagi-source.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            "Expected exactly one kusanagi-source.zip, found "
            f"{[str(path) for path in archives]}"
        )
    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    CODE_DIR.mkdir(parents=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(CODE_DIR)
    if not (CODE_DIR / "train.py").is_file():
        raise RuntimeError("Source archive does not contain train.py at root")
    return CODE_DIR


def locate_or_extract_bonsai() -> tuple[Path, bool]:
    csv_candidates = [
        path
        for path in INPUT_ROOT.rglob("test_poses.csv")
        if path.parent.name == "test" and path.parent.parent.name == "bonsai"
    ]
    if len(csv_candidates) == 1:
        return csv_candidates[0].parent.parent, False
    if len(csv_candidates) > 1:
        raise RuntimeError(
            "Multiple bonsai test_poses.csv files found: "
            f"{[str(path) for path in csv_candidates]}"
        )

    archives = list(INPUT_ROOT.rglob("bonsai.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            "Expected exactly one bonsai.zip, found "
            f"{[str(path) for path in archives]}"
        )
    if EXTRACTED_DATA_DIR.exists():
        shutil.rmtree(EXTRACTED_DATA_DIR)
    EXTRACTED_DATA_DIR.mkdir(parents=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(EXTRACTED_DATA_DIR)
    extracted_csv = list(EXTRACTED_DATA_DIR.rglob("test_poses.csv"))
    if len(extracted_csv) != 1:
        raise RuntimeError(
            "Unable to uniquely locate extracted bonsai/test/test_poses.csv"
        )
    return extracted_csv[0].parent.parent, True


def install_runtime(source_dir: Path) -> None:
    def module_available(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            return False

    required = {
        "plyfile": "plyfile",
        "einops": "einops",
        "wandb": "wandb",
        "cv2": "opencv-python-headless",
        "colorama": "colorama",
    }
    missing = [
        package
        for module, package in required.items()
        if not module_available(module)
    ]
    if missing:
        run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            "INSTALL PYTHON DEPENDENCIES",
        )

    os.environ.setdefault("MAX_JOBS", "2")
    for relative_path, module in (
        ("submodules/simple-knn", "simple_knn._C"),
        (
            "submodules/diff-gaussian-rasterization",
            "diff_gaussian_rasterization._C",
        ),
    ):
        if not module_available(module):
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--no-build-isolation",
                    source_dir / relative_path,
                ],
                f"BUILD {relative_path}",
            )


def read_expected(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_name", "width", "height"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(
            f"{csv_path} must contain columns {sorted(required)}"
        )
    names = [row["image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate image_name values in test_poses.csv")
    return rows


def create_validated_zip(
    render_dir: Path,
    csv_path: Path,
) -> dict[str, object]:
    rows = read_expected(csv_path)
    expected_names = [row["image_name"] for row in rows]
    expected_set = set(expected_names)
    actual_files = [path for path in render_dir.iterdir() if path.is_file()]
    actual_by_name = {path.name: path for path in actual_files}
    actual_set = set(actual_by_name)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise RuntimeError(
            f"Render/CSV mismatch. missing={missing}, extra={extra}"
        )

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    scene_dir = PACKAGE_DIR / "bonsai"
    scene_dir.mkdir(parents=True)
    image_records = []
    for row in rows:
        name = row["image_name"]
        expected_size = (int(row["width"]), int(row["height"]))
        source = actual_by_name[name]
        destination = scene_dir / name
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.size != expected_size:
                raise RuntimeError(
                    f"{name}: got {image.size}, expected {expected_size}"
                )
            image.save(
                destination,
                format="JPEG",
                quality=100,
                subsampling=0,
                optimize=False,
            )
        with Image.open(destination) as verified:
            if verified.format != "JPEG" or verified.size != expected_size:
                raise RuntimeError(f"Invalid packaged JPEG: {destination}")
            verified.verify()
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        image_records.append(
            {
                "name": name,
                "archive_path": f"bonsai/{name}",
                "width": expected_size[0],
                "height": expected_size[1],
                "sha256": digest,
            }
        )

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_STORED) as archive:
        for record in image_records:
            archive.write(
                PACKAGE_DIR / str(record["archive_path"]),
                arcname=str(record["archive_path"]),
            )

    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        corrupt = archive.testzip()
        archived_names = archive.namelist()
    expected_archive_names = [
        f"bonsai/{name}" for name in expected_names
    ]
    if corrupt is not None or archived_names != expected_archive_names:
        raise RuntimeError(
            f"ZIP validation failed: corrupt={corrupt}, "
            f"names_match={archived_names == expected_archive_names}"
        )

    return {
        "scene": "bonsai",
        "iterations": ITERATIONS,
        "csv": str(csv_path),
        "image_count": len(image_records),
        "archive_order_matches_csv": True,
        "jpeg_quality": 100,
        "jpeg_subsampling": "4:4:4",
        "zip_sha256": hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest(),
        "images": image_records,
    }


def main() -> None:
    run(["nvidia-smi"], "GPU PREFLIGHT")
    source_dir = locate_or_extract_source()
    bonsai_dir, remove_extracted_data = locate_or_extract_bonsai()
    csv_path = bonsai_dir / "test" / "test_poses.csv"
    print("Source:", source_dir)
    print("Bonsai:", bonsai_dir)
    print("CSV rows:", len(read_expected(csv_path)))

    install_runtime(source_dir)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    common_model_args = [
        "-s",
        bonsai_dir,
        "-m",
        MODEL_DIR,
        "-r",
        "1",
        "--data_device",
        "cpu",
        "--appearance_dim",
        "0",
    ]
    train_command = [
        sys.executable,
        "train.py",
        *common_model_args,
        "--gpu",
        "0",
        "--iterations",
        str(ITERATIONS),
        "--save_iterations",
        str(ITERATIONS),
        "--test_iterations",
        "-1",
        "--use_gaussianpro",
        "--gaussianpro_start_iter",
        "1000",
        "--gaussianpro_until_iter",
        "7000",
        "--gaussianpro_interval",
        "50",
        "--gaussianpro_neighbors",
        "4",
        "--gaussianpro_downsample",
        "4",
        "--gaussianpro_patch_radius",
        "2",
        "--gaussianpro_patchmatch_iterations",
        "3",
        "--gaussianpro_min_consistent_views",
        "2",
        "--gaussianpro_max_anchors_per_step",
        "512",
        "--gaussianpro_voxel_factor",
        "0.75",
    ]
    run(train_command, "TRAIN BONSAI GAUSSIANPRO 7000", cwd=source_dir)

    render_command = [
        sys.executable,
        "render.py",
        *common_model_args,
        "--iteration",
        str(ITERATIONS),
        "--eval",
        "--skip_train",
    ]
    run(render_command, "RENDER BONSAI TEST POSES", cwd=source_dir)
    render_dir = (
        MODEL_DIR
        / "test"
        / f"ours_{ITERATIONS}"
        / "renders"
    )
    if not render_dir.is_dir():
        raise RuntimeError(f"Missing render directory: {render_dir}")

    manifest = create_validated_zip(render_dir, csv_path)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "images"}, indent=2))

    # Avoid publishing the model/source as kernel output; keep only the ZIP
    # and compact verification manifest requested by the user.
    shutil.rmtree(MODEL_DIR)
    shutil.rmtree(PACKAGE_DIR)
    if source_dir == CODE_DIR and CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    if remove_extracted_data and EXTRACTED_DATA_DIR.exists():
        shutil.rmtree(EXTRACTED_DATA_DIR)
    print(f"DONE: {OUTPUT_ZIP}")
    print(f"MANIFEST: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
