import argparse
import shutil
import zipfile
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--quality", type=int, required=True)
    parser.add_argument("--subsampling", type=int, default=0)
    args = parser.parse_args()

    source_files = sorted(path for path in args.source.rglob("*") if path.is_file())
    if not source_files:
        raise RuntimeError(f"No files found under {args.source}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    source_sizes = {}
    for index, source_path in enumerate(source_files, 1):
        relative_path = source_path.relative_to(args.source)
        output_path = args.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            image.load()
            source_sizes[relative_path] = image.size
            image.convert("RGB").save(
                output_path,
                format="JPEG",
                quality=args.quality,
                subsampling=args.subsampling,
                optimize=True,
            )
        if index % 50 == 0 or index == len(source_files):
            print(f"encoded {index}/{len(source_files)}")

    if args.output_zip.exists():
        args.output_zip.unlink()
    with zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for output_path in sorted(args.output_dir.rglob("*")):
            if output_path.is_file():
                archive.write(
                    output_path,
                    arcname=(
                        Path(args.source.name)
                        / output_path.relative_to(args.output_dir)
                    ).as_posix(),
                )

    with zipfile.ZipFile(args.output_zip, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        archived_files = [name for name in archive.namelist() if not name.endswith("/")]

    output_files = sorted(path for path in args.output_dir.rglob("*") if path.is_file())
    if len(output_files) != len(source_files) or len(archived_files) != len(source_files):
        raise RuntimeError("File count changed during repack")
    for output_path in output_files:
        relative_path = output_path.relative_to(args.output_dir)
        with Image.open(output_path) as image:
            if image.size != source_sizes[relative_path]:
                raise RuntimeError(f"Image dimensions changed: {relative_path}")
            image.verify()

    print(
        f"quality={args.quality} subsampling={args.subsampling} "
        f"files={len(source_files)} zip_mib={args.output_zip.stat().st_size / 1024**2:.2f}"
    )


if __name__ == "__main__":
    main()
