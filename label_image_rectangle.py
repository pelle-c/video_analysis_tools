import argparse
import sys
from pathlib import Path

import cv2


WINDOW_NAME = "Draw label - drag rectangle, Enter/s to save, r to reset, q delete, Esc quit"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw one rectangle on an image, or every unlabeled JPEG in a folder, "
            "and save YOLO label files."
        )
    )
    parser.add_argument("path", help="Path to an image or a folder of JPEG images.")
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Default YOLO class ID when saving. Default: 0.",
    )
    parser.add_argument(
        "--classes",
        default="cat,dog,robot_mower,person,car",
        help="Comma-separated class names shown when saving. Default: cat,dog,robot_mower,person.",
    )
    parser.add_argument(
        "--output",
        help="Output label file for single-image mode. Default: same path as image, with .txt extension.",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=1200,
        help="Maximum display width or height. Default: 1200.",
    )
    return parser.parse_args()


def parse_classes(classes_text):
    return [class_name.strip() for class_name in classes_text.split(",") if class_name.strip()]


def find_unlabeled_images(folder_path):
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        image_paths.extend(folder_path.glob(pattern))

    return [
        image_path
        for image_path in sorted(set(image_paths))
        if not image_path.with_suffix(".txt").exists()
    ]


def get_display_scale(image, max_window_size):
    height, width = image.shape[:2]
    largest_side = max(width, height)
    if largest_side <= max_window_size:
        return 1.0
    return max_window_size / largest_side


def normalize_box(start_point, end_point, image_shape, display_scale):
    height, width = image_shape[:2]
    x1 = int(round(min(start_point[0], end_point[0]) / display_scale))
    y1 = int(round(min(start_point[1], end_point[1]) / display_scale))
    x2 = int(round(max(start_point[0], end_point[0]) / display_scale))
    y2 = int(round(max(start_point[1], end_point[1]) / display_scale))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def box_to_yolo_line(box, image_shape, class_id):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2

    return (
        f"{class_id} "
        f"{x_center / width:.6f} "
        f"{y_center / height:.6f} "
        f"{box_width / width:.6f} "
        f"{box_height / height:.6f}\n"
    )


def draw_label(image_path, output_path, default_class_id, class_names, max_window_size):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not open image '{image_path}'.")
        return 1

    display_scale = get_display_scale(image, max_window_size)
    if display_scale != 1.0:
        display = cv2.resize(
            image,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = image.copy()

    state = {
        "drawing": False,
        "start": None,
        "current": None,
        "box": None,
        "choosing_class": False,
    }

    def redraw():
        canvas = display.copy()
        start = state["start"]
        current = state["current"]

        if start and current:
            cv2.rectangle(canvas, start, current, (0, 255, 0), 2)

        if state["box"]:
            x1, y1, x2, y2 = state["box"]
            scaled_box = (
                int(x1 * display_scale),
                int(y1 * display_scale),
                int(x2 * display_scale),
                int(y2 * display_scale),
            )
            cv2.rectangle(
                canvas,
                (scaled_box[0], scaled_box[1]),
                (scaled_box[2], scaled_box[3]),
                (0, 255, 0),
                2,
            )

        if state["choosing_class"]:
            overlay = canvas.copy()
            panel_height = 70 + 28 * max(1, min(len(class_names), 10))
            cv2.rectangle(
                overlay,
                (10, 10),
                (min(canvas.shape[1] - 10, 520), min(canvas.shape[0] - 10, panel_height)),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
            cv2.putText(
                canvas,
                "Choose class: press number, Enter=default, r=back",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for class_id, class_name in enumerate(class_names[:10]):
                default_marker = " default" if class_id == default_class_id else ""
                cv2.putText(
                    canvas,
                    f"{class_id}: {class_name}{default_marker}",
                    (24, 76 + class_id * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if class_id == default_class_id else (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
            state["box"] = None
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            state["current"] = (x, y)
            box = normalize_box(
                state["start"],
                state["current"],
                image.shape,
                display_scale,
            )
            state["box"] = box
            state["start"] = None
            state["current"] = None
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("No label saved. Image marked for deletion.")
            return "delete"
        if key == 27:
            cv2.destroyAllWindows()
            print("No label saved.")
            return "quit"

        if state["choosing_class"]:
            if key == ord("r"):
                state["choosing_class"] = False
                redraw()
                continue
            if key == 13:
                chosen_class_id = default_class_id
            elif ord("0") <= key <= ord("9"):
                chosen_class_id = key - ord("0")
                if class_names and chosen_class_id >= len(class_names):
                    print(f"Class {chosen_class_id} is not in the configured class list.")
                    continue
            else:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                box_to_yolo_line(state["box"], image.shape, chosen_class_id),
                encoding="utf-8",
            )
            cv2.destroyAllWindows()
            class_name = (
                class_names[chosen_class_id]
                if chosen_class_id < len(class_names)
                else str(chosen_class_id)
            )
            print(f"Saved YOLO label to '{output_path}' as class {chosen_class_id}: {class_name}.")
            return "saved"

        if key == ord("r"):
            state["box"] = None
            state["start"] = None
            state["current"] = None
            state["drawing"] = False
            state["choosing_class"] = False
            redraw()
        if key in (ord("s"), 13):
            if not state["box"]:
                print("Draw a rectangle before saving.")
                continue

            state["choosing_class"] = True
            redraw()


def label_single_image(image_path, output_path, class_id, class_names, max_window_size):
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' was not found.")
        return 1

    result = draw_label(image_path, output_path, class_id, class_names, max_window_size)
    return 0 if result == "saved" else 1


def label_folder(folder_path, class_id, class_names, max_window_size):
    image_paths = find_unlabeled_images(folder_path)
    if not image_paths:
        print(f"No unlabeled JPEG images found in '{folder_path}'.")
        return 0

    print(f"Found {len(image_paths)} unlabeled JPEG image(s).")
    saved_count = 0
    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] Labeling {image_path}")
        result = draw_label(
            image_path,
            image_path.with_suffix(".txt"),
            class_id,
            class_names,
            max_window_size,
        )
        if result == "saved":
            saved_count += 1
            continue
        if result == "delete":
            image_path.unlink()
            print(f"Deleted image '{image_path}'.")
            continue
        if result == "quit":
            break

    print(f"Saved {saved_count} label file(s).")
    return 0


def main():
    args = parse_args()
    class_names = parse_classes(args.classes)
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' was not found.")
        return 1

    if path.is_dir():
        if args.output:
            print("Warning: --output is ignored when labeling a folder.")
        return label_folder(path, args.class_id, class_names, args.max_window_size)

    output_path = Path(args.output) if args.output else path.with_suffix(".txt")
    return label_single_image(
        path,
        output_path,
        args.class_id,
        class_names,
        args.max_window_size,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nLabeling interrupted.")
        sys.exit(130)
