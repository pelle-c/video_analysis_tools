import argparse
import re
from pathlib import Path


DEFAULT_DATA_YAML = "dataset/data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename YOLO label files and matching JPEGs to start with the class name."
        )
    )
    parser.add_argument("folder", help="Folder containing YOLO .txt label files.")
    parser.add_argument(
        "--data-yaml",
        default=DEFAULT_DATA_YAML,
        help=f"YOLO data.yaml with class names. Default: {DEFAULT_DATA_YAML}.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process label files in subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing files.",
    )
    return parser.parse_args()


def safe_prefix(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_class_names(data_yaml_path):
    names = {}
    in_names = False

    for line in data_yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "names:":
            in_names = True
            continue

        if in_names and not line.startswith((" ", "\t")):
            break

        if not in_names:
            continue

        match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if match:
            class_id = int(match.group(1))
            class_name = match.group(2).strip().strip("\"'")
            names[class_id] = safe_prefix(class_name)

    if not names:
        raise ValueError(f"No class names found in '{data_yaml_path}'.")

    return names


def read_label_class_id(label_path):
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        first_value = stripped.split(maxsplit=1)[0]
        try:
            return int(first_value)
        except ValueError as exc:
            raise ValueError(f"Invalid class ID in '{label_path}': {first_value}") from exc

    raise ValueError(f"No labels found in '{label_path}'.")


def find_matching_jpeg(label_path):
    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        image_path = label_path.with_suffix(suffix)
        if image_path.exists():
            return image_path
    return None


def prefixed_path(path, class_name, class_prefixes):
    if any(path.name.startswith(f"{prefix}_") for prefix in class_prefixes):
        return path
    return path.with_name(f"{class_name}_{path.name}")


def rename_pair(label_path, image_path, class_name, class_prefixes, dry_run):
    new_label_path = prefixed_path(label_path, class_name, class_prefixes)
    new_image_path = (
        prefixed_path(image_path, class_name, class_prefixes) if image_path else None
    )

    planned = [(label_path, new_label_path)]
    if image_path:
        planned.append((image_path, new_image_path))

    planned = [(source, target) for source, target in planned if source != target]

    if not planned:
        return "skipped"

    for source, target in planned:
        if target.exists():
            raise FileExistsError(f"Target already exists: {target}")

    for source, target in planned:
        action = "Would rename" if dry_run else "Renamed"
        print(f"{action}: {source} -> {target}")
        if not dry_run:
            source.rename(target)

    return "renamed"


def main():
    args = parse_args()
    folder_path = Path(args.folder)
    data_yaml_path = Path(args.data_yaml)

    if not folder_path.is_dir():
        print(f"Error: Folder '{folder_path}' was not found.")
        return 1

    if not data_yaml_path.exists():
        print(f"Error: data.yaml '{data_yaml_path}' was not found.")
        return 1

    try:
        class_names = load_class_names(data_yaml_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    pattern = "**/*.txt" if args.recursive else "*.txt"
    label_paths = sorted(folder_path.glob(pattern))
    class_prefixes = set(class_names.values())
    renamed_count = 0
    skipped_count = 0

    for label_path in label_paths:
        try:
            class_id = read_label_class_id(label_path)
            class_name = class_names[class_id]
            image_path = find_matching_jpeg(label_path)
            if image_path is None:
                print(f"Skipped missing image for: {label_path}")
                skipped_count += 1
                continue

            result = rename_pair(
                label_path,
                image_path,
                class_name,
                class_prefixes,
                args.dry_run,
            )
        except KeyError:
            print(f"Skipped unknown class ID in: {label_path}")
            skipped_count += 1
            continue
        except (UnicodeDecodeError, ValueError, FileExistsError) as exc:
            print(f"Skipped {label_path}: {exc}")
            skipped_count += 1
            continue

        if result == "renamed":
            renamed_count += 1
        else:
            skipped_count += 1

    action = "would rename" if args.dry_run else "renamed"
    print(f"Done: {action} {renamed_count} pair(s), skipped {skipped_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
