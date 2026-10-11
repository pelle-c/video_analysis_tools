import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Change the class ID for every YOLO label file in a folder."
    )
    parser.add_argument("folder", help="Folder containing YOLO .txt label files.")
    parser.add_argument(
        "class_id",
        type=int,
        help="Class ID to write as the first value on every label line.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also update label files in subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would change without writing them.",
    )
    return parser.parse_args()


def find_label_files(folder_path, recursive):
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(folder_path.glob(pattern))


def update_label_text(text, class_id):
    updated_lines = []
    changed = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            updated_lines.append(line)
            continue

        line_ending = ""
        content = line
        if content.endswith("\r\n"):
            line_ending = "\r\n"
            content = content[:-2]
        elif content.endswith("\n"):
            line_ending = "\n"
            content = content[:-1]

        parts = content.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Expected YOLO line with class and coordinates: {line!r}")

        updated_line = f"{class_id} {parts[1]}{line_ending}"
        updated_lines.append(updated_line)
        changed = changed or updated_line != line

    return "".join(updated_lines), changed


def main():
    args = parse_args()
    folder_path = Path(args.folder)

    if args.class_id < 0:
        print("Error: class_id must be 0 or greater.")
        return 1

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' was not found.")
        return 1

    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a folder.")
        return 1

    label_files = find_label_files(folder_path, args.recursive)
    if not label_files:
        print(f"No .txt label files found in '{folder_path}'.")
        return 0

    changed_count = 0
    skipped_count = 0

    for label_path in label_files:
        try:
            original_text = label_path.read_text()
            updated_text, changed = update_label_text(original_text, args.class_id)
        except UnicodeDecodeError:
            print(f"Skipped unreadable file: {label_path}")
            skipped_count += 1
            continue
        except ValueError as exc:
            print(f"Skipped malformed label file '{label_path}': {exc}")
            skipped_count += 1
            continue

        if not changed:
            skipped_count += 1
            continue

        changed_count += 1
        if args.dry_run:
            print(f"Would update: {label_path}")
        else:
            label_path.write_text(updated_text)
            print(f"Updated: {label_path}")

    action = "would update" if args.dry_run else "updated"
    print(f"Done: {action} {changed_count} file(s), skipped {skipped_count} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
