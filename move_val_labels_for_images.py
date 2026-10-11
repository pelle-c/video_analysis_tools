import argparse
import shutil
from pathlib import Path


DEFAULT_IMAGE_DIR = "dataset/images/val"
DEFAULT_SOURCE_LABEL_DIR = "dataset/labels/train"
DEFAULT_OUTPUT_DIR = "dataset/labels/val"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For every JPEG in dataset/images/val, move the matching YOLO label "
            "file from dataset/labels/train to dataset/labels/val."
        )
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help=f"Folder containing validation JPEGs. Default: {DEFAULT_IMAGE_DIR}.",
    )
    parser.add_argument(
        "--source-label-dir",
        default=DEFAULT_SOURCE_LABEL_DIR,
        help=(
            "Folder containing labels to move. "
            f"Default: {DEFAULT_SOURCE_LABEL_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to move labels into. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without changing files.",
    )
    return parser.parse_args()


def find_jpegs(image_dir):
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        image_paths.extend(image_dir.glob(pattern))
    return sorted(set(image_paths))


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    source_label_dir = Path(args.source_label_dir)
    output_dir = Path(args.output_dir)

    if not image_dir.is_dir():
        print(f"Error: image-dir '{image_dir}' was not found.")
        return 1

    if not source_label_dir.is_dir():
        print(f"Error: source-label-dir '{source_label_dir}' was not found.")
        return 1

    image_paths = find_jpegs(image_dir)
    if not image_paths:
        print(f"No JPEG files found in '{image_dir}'.")
        return 0

    moved_count = 0
    missing_count = 0
    skipped_count = 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        source_label_path = source_label_dir / f"{image_path.stem}.txt"
        target_label_path = output_dir / source_label_path.name

        if not source_label_path.exists():
            print(f"Missing label: {source_label_path}")
            missing_count += 1
            continue

        if source_label_path == target_label_path:
            print(f"Already in place: {source_label_path}")
            skipped_count += 1
            continue

        if target_label_path.exists():
            print(f"Skipped existing target: {target_label_path}")
            skipped_count += 1
            continue

        action = "Would move" if args.dry_run else "Moved"
        print(f"{action}: {source_label_path} -> {target_label_path}")
        if not args.dry_run:
            shutil.move(str(source_label_path), str(target_label_path))
        moved_count += 1

    action = "would move" if args.dry_run else "moved"
    print(
        f"Done: {action} {moved_count} label file(s), "
        f"missing {missing_count}, skipped {skipped_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
