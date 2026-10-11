import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import cv2


DEFAULT_MODEL = "yolov8s.pt"
# DEFAULT_MODEL = "runs/detect/train-6/weights/best.pt"
ANALYZED_LIST_FILENAME = "analyzed_videos.txt"
DATASET_CLASSES = ["cat",  "dog", "robot_mower", "person","car"]
CLASS_ALIASES = {
    "automower": "robot_mower",
    "robot mower": "robot_mower",
    "robotic lawn mower": "robot_mower",
    "robotgrasklippare": "robot_mower",
    "robotgräsklippare": "robot_mower",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Save JPEG frames, annotated JPEGs, and YOLO labels for every new "
            "MP4 in a raw folder."
        )
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Analyze one specific MP4 file instead of scanning raw-dir.",
    )
    parser.add_argument(
        "--raw-dir",
        default="raw",
        help="Folder containing MP4 files. Default: raw.",
    )
    parser.add_argument(
        "--output-dir",
        default="raw_analyzed",
        help="Folder where JPEGs and label files are saved. Default: raw_analyzed.",
    )
    parser.add_argument(
        "--second-check-dir",
        default="raw_second_check",
        help="Folder where MP4s with no saved detections are copied. Default: raw_second_check.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model path or name. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--second-model",
        help=(
            "YOLO model path or name to try if the first model saves no detections. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence. Default: 0.35.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size. Default: 960.",
    )
    parser.add_argument(
        "--images-per-video",
        type=int,
        default=10,
        help="Maximum detected frames to save per video. Default: 10.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=15,
        help="Minimum frames to skip after saving a detection. Default: 15.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subfolders inside raw-dir.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DATASET_CLASSES),
        help=(
            "Comma-separated dataset classes to export in label files. "
            "Default: cat,person,car,dog,mower,robot_mower."
        ),
    )
    return parser.parse_args()


def load_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("Error: The 'ultralytics' package is not installed.")
        print("Install it with: python -m pip install ultralytics")
        raise SystemExit(1) from exc

    try:
        return YOLO(model_path)
    except Exception as exc:
        print(f"Error: Could not load YOLO model '{model_path}'.")
        print(f"Details: {exc}")
        raise SystemExit(1) from exc


def safe_filename(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_class_name(value):
    normalized = value.lower().replace("_", " ").replace("-", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return CLASS_ALIASES.get(normalized, normalized)


def parse_dataset_classes(classes_text):
    classes = [
        normalize_class_name(class_name)
        for class_name in classes_text.split(",")
        if class_name.strip()
    ]
    if not classes:
        print("Error: At least one class must be configured.")
        raise SystemExit(1)
    return {class_name: index for index, class_name in enumerate(classes)}


def find_videos(raw_dir, recursive):
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"Error: Raw folder '{raw_dir}' was not found.")
        raise SystemExit(1)

    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted(raw_path.glob(pattern))


def get_videos(args):
    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            print(f"Error: MP4 file '{video_path}' was not found.")
            raise SystemExit(1)
        if video_path.suffix.lower() != ".mp4":
            print(f"Error: '{video_path}' is not an MP4 file.")
            raise SystemExit(1)
        return [video_path]

    return find_videos(args.raw_dir, args.recursive)


def get_analyzed_list_path(output_dir):
    return Path(output_dir) / ANALYZED_LIST_FILENAME


def load_analyzed_video_keys(analyzed_list_path):
    if not analyzed_list_path.exists():
        return set()

    return {
        line.strip()
        for line in analyzed_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def write_analyzed_video_keys(analyzed_list_path, analyzed_video_keys):
    analyzed_list_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sorted(analyzed_video_keys))
    analyzed_list_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def migrate_done_files(output_dir, analyzed_list_path):
    output_path = Path(output_dir)
    analyzed_video_keys = load_analyzed_video_keys(analyzed_list_path)
    done_keys = {done_path.stem for done_path in output_path.glob("*.done")}
    new_keys = done_keys - analyzed_video_keys

    if new_keys:
        analyzed_video_keys.update(new_keys)
        write_analyzed_video_keys(analyzed_list_path, analyzed_video_keys)
        print(
            f"Added {len(new_keys)} existing .done marker(s) to "
            f"'{analyzed_list_path}'."
        )

    return analyzed_video_keys


def get_dataset_class_id(box, names, class_ids_by_name):
    model_class_id = int(box.cls)
    model_class_name = names.get(model_class_id, f"class {model_class_id}")
    class_name = normalize_class_name(model_class_name)
    return class_ids_by_name.get(class_name)


def get_display_name(box, names):
    model_class_id = int(box.cls)
    model_class_name = names.get(model_class_id, f"class {model_class_id}")
    return normalize_class_name(model_class_name).title()


def draw_detections(frame, boxes, names):
    annotated = frame.copy()

    for box in boxes:
        confidence = float(box.conf[0])
        object_name = get_display_name(box, names)
        label = f"{object_name} {confidence:.0%}"

        height, width = annotated.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        text_width, text_height = text_size
        label_y = max(y1, text_height + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_height - baseline - 4),
            (x1 + text_width + 6, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def box_to_yolo_label(box, frame_shape, names, class_ids_by_name):
    height, width = frame_shape[:2]
    label_class_id = get_dataset_class_id(box, names, class_ids_by_name)
    if label_class_id is None:
        return None

    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2

    return (
        f"{label_class_id} "
        f"{x_center / width:.6f} "
        f"{y_center / height:.6f} "
        f"{box_width / width:.6f} "
        f"{box_height / height:.6f}"
    )


def write_label_file(label_path, boxes, frame_shape, names, class_ids_by_name):
    labels = []
    for box in boxes:
        label = box_to_yolo_label(box, frame_shape, names, class_ids_by_name)
        if label:
            labels.append(label)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def scan_video(video_path, model, args, class_ids_by_name):
    output_dir = Path(args.output_dir)
    video_key = safe_filename(video_path.stem)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open '{video_path}', skipping.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_number = 0
    skip_until_frame = 0

    try:
        while saved_count < args.images_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number < skip_until_frame:
                continue

            results = model(frame, imgsz=args.imgsz, verbose=False)
            boxes = []
            for result in results:
                boxes.extend(
                    box
                    for box in result.boxes
                    if float(box.conf[0]) >= args.confidence
                    and get_dataset_class_id(box, model.names, class_ids_by_name)
                    is not None
                )

            if not boxes:
                continue

            annotated = draw_detections(frame, boxes, model.names)
            saved_count += 1
            best_confidence = max(float(box.conf[0]) for box in boxes)
            file_stem = (
                f"{video_key}_frame-{frame_number:06d}"
                f"_det-{saved_count:02d}_conf-{best_confidence:.2f}"
            )
            frame_path = output_dir / f"{file_stem}.jpg"
            annotated_path = output_dir / f"{file_stem}_annotated.jpg"
            label_path = output_dir / f"{file_stem}.txt"

            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(annotated_path), annotated)
            write_label_file(
                label_path,
                boxes,
                frame.shape,
                model.names,
                class_ids_by_name,
            )

            skip_until_frame = frame_number + args.frame_gap
    finally:
        cap.release()

    return saved_count


def annotate_video(
    video_path,
    model,
    second_model,
    args,
    class_ids_by_name,
    analyzed_video_keys,
    analyzed_list_path,
):
    output_dir = Path(args.output_dir)
    video_key = safe_filename(video_path.stem)
    if video_key in analyzed_video_keys:
        print(f"Skipping {video_path}; already analyzed.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = scan_video(video_path, model, args, class_ids_by_name)
    if saved_count == 0 and second_model is not None:
        print(
            f"  No detections saved with '{args.model}'. "
            f"Trying '{args.second_model}'..."
        )
        saved_count = scan_video(video_path, second_model, args, class_ids_by_name)

    if saved_count == 0:
        print(f"Warning: No matching detections were saved from '{video_path}'.")
        second_check_dir = Path(args.second_check_dir)
        second_check_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, second_check_dir / video_path.name)

    analyzed_video_keys.add(video_key)
    write_analyzed_video_keys(analyzed_list_path, analyzed_video_keys)
    return saved_count


def main():
    args = parse_args()
    videos = get_videos(args)
    if not videos:
        print(f"No MP4 files found in '{args.raw_dir}'.")
        return 1

    model = load_model(args.model)
    second_model = load_model(args.second_model) if args.second_model else None
    class_ids_by_name = parse_dataset_classes(args.classes)
    os.makedirs(args.output_dir, exist_ok=True)
    analyzed_list_path = get_analyzed_list_path(args.output_dir)
    analyzed_video_keys = migrate_done_files(args.output_dir, analyzed_list_path)

    total_saved = 0
    analyzed_count = 0
    skipped_count = 0
    for video_path in videos:
        print(f"Annotating {video_path}...")
        saved_count = annotate_video(
            video_path,
            model,
            second_model,
            args,
            class_ids_by_name,
            analyzed_video_keys,
            analyzed_list_path,
        )
        if saved_count is None:
            skipped_count += 1
            continue

        analyzed_count += 1
        total_saved += saved_count
        print(f"  Saved {saved_count} analyzed frame set(s).")

    print(
        f"\nDone. Saved {total_saved} analyzed frame set(s) from "
        f"{analyzed_count} analyzed video(s) to '{args.output_dir}'."
    )
    if skipped_count:
        print(f"Skipped {skipped_count} already analyzed video(s).")
    print("Dataset class IDs:")
    for class_name, class_id in class_ids_by_name.items():
        print(f"  {class_id}: {class_name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAnnotation interrupted.")
        sys.exit(130)
