import argparse
import multiprocessing
import os
import re
import signal
import sys

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect objects in an MP4 video with a YOLO model."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Path to the MP4 video to analyze.",
    )
    parser.add_argument(
        "--max-bad-frames",
        type=int,
        default=25,
        help="How many unreadable frames to skip before giving up. Default: 25.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence to count an object. Default: 0.45.",
    )
    parser.add_argument(
        "--min-object-hits",
        type=int,
        default=3,
        help="Minimum detections needed before reporting an object type. Default: 3.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path or name. Try yolov8s.pt or yolov8m.pt for better accuracy.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size. Larger values can help small objects. Default: 960.",
    )
    parser.add_argument(
        "--annotated-dir",
        default="annotated_objects",
        help="Directory for annotated object images. Default: annotated_objects.",
    )
    parser.add_argument(
        "--no-save-annotated",
        action="store_true",
        help="Do not save one annotated JPG per detected object type.",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Run analysis in the main process instead of a protected child process.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small YOLO test without reading a video.",
    )
    return parser.parse_args()


def load_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("Error: The 'ultralytics' package is not installed.")
        print("Install it with: pip install ultralytics")
        raise SystemExit(1) from exc

    try:
        return YOLO(model_path)
    except Exception as exc:
        print("Error: Could not load the YOLO model.")
        print(f"Details: {exc}")
        raise SystemExit(1) from exc


def open_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' was not found.")
        raise SystemExit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open or read video file: {video_path}")
        raise SystemExit(1)

    return cap


def read_next_frame(cap, bad_frames, max_bad_frames):
    ret, frame = cap.read()
    if ret:
        return frame, bad_frames

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or current_frame >= total_frames:
        return None, bad_frames

    bad_frames += 1
    if bad_frames > max_bad_frames:
        print(
            "Error: Too many unreadable video frames. The input video may be corrupt."
        )
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)
    return False, bad_frames


def safe_filename(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def get_model_class_name(model, class_id):
    return model.names.get(class_id, f"class_{class_id}")


def save_annotated_object(
    frame,
    box,
    object_name,
    video_key,
    model_key,
    frame_number,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    confidence = float(box.conf[0])
    label = f"{object_name} {confidence:.0%}"
    color = (0, 255, 0)

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    text_width, text_height = text_size
    label_y = max(y1, text_height + baseline + 4)
    cv2.rectangle(
        annotated_frame,
        (x1, label_y - text_height - baseline - 4),
        (x1 + text_width + 6, label_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        annotated_frame,
        label,
        (x1 + 3, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    filename = (
        f"{video_key}_{safe_filename(object_name)}_frame-{frame_number:06d}"
        f"_conf-{confidence:.2f}_model-{model_key}.jpg"
    )
    output_path = os.path.join(output_dir, filename)
    if not cv2.imwrite(output_path, annotated_frame):
        return None

    return output_path


def analyze_video(args):
    video_path = args.video_path
    video_key = safe_filename(os.path.splitext(os.path.basename(video_path))[0])
    model_key = safe_filename(args.model)
    cap = open_video(video_path)
    model = load_model(args.model)
    object_counts = {}
    object_confidences = {}
    bad_frames = 0
    annotated_paths = {}

    print(f"Analyzing '{video_path}'... Please wait...")

    try:
        while cap.isOpened():
            frame, bad_frames = read_next_frame(
                cap, bad_frames, args.max_bad_frames
            )
            if frame is None:
                break
            if frame is False:
                continue

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            results = model(frame, imgsz=args.imgsz, verbose=False)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf[0])
                    if confidence < args.confidence:
                        continue

                    object_name = get_model_class_name(model, class_id)

                    object_counts[object_name] = object_counts.get(object_name, 0) + 1
                    object_confidences[object_name] = max(
                        object_confidences.get(object_name, 0.0), confidence
                    )

                    if (
                        not args.no_save_annotated
                        and object_name not in annotated_paths
                        and object_counts[object_name] >= args.min_object_hits
                    ):
                        output_path = save_annotated_object(
                            frame,
                            box,
                            object_name,
                            video_key,
                            model_key,
                            frame_number,
                            args.annotated_dir,
                        )
                        if output_path:
                            annotated_paths[object_name] = output_path

    finally:
        cap.release()

    detected_objects = {
        object_name
        for object_name in object_counts
        if object_counts.get(object_name, 0) >= args.min_object_hits
    }

    return {
        "detected_objects": detected_objects,
        "object_counts": object_counts,
        "object_confidences": object_confidences,
        "annotated_paths": annotated_paths,
        "annotated_dir": args.annotated_dir,
    }


def self_test():
    try:
        import numpy as np
    except ImportError as exc:
        print("Error: The 'numpy' package is not installed.")
        raise SystemExit(1) from exc

    model = load_model("yolov8n.pt")
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model(test_frame, verbose=False)
    print("Self-test passed: YOLO loaded and analyzed a blank frame.")


def print_result(analysis_result):
    detected_objects = analysis_result["detected_objects"]
    object_counts = analysis_result["object_counts"]
    object_confidences = analysis_result["object_confidences"]
    annotated_paths = analysis_result["annotated_paths"]

    print("\n--- ANALYSIS COMPLETE ---")
    if detected_objects:
        print("Objects detected in the video:")
        for object_name in sorted(detected_objects):
            print(
                f"  [X] {object_name} "
                f"({object_counts[object_name]} hits, "
                f"max {object_confidences[object_name]:.0%})"
            )
    else:
        print("No objects met the detection threshold.")

    if annotated_paths:
        print(
            f"Saved {len(annotated_paths)} annotated object image(s) to "
            f"'{analysis_result['annotated_dir']}'."
        )


def run_worker(result_queue, args):
    try:
        if args.self_test:
            self_test()
            result_queue.put({"ok": True, "detected_objects": []})
            return

        analysis_result = analyze_video(args)
        result_queue.put(
            {
                "ok": True,
                "detected_objects": sorted(analysis_result["detected_objects"]),
                "object_counts": analysis_result["object_counts"],
                "object_confidences": analysis_result["object_confidences"],
                "annotated_paths": analysis_result["annotated_paths"],
                "annotated_dir": analysis_result["annotated_dir"],
            }
        )
    except SystemExit as exc:
        result_queue.put({"ok": False, "exit_code": exc.code})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def print_native_crash_help(exit_code):
    signal_number = -exit_code
    signal_name = signal.Signals(signal_number).name

    print(f"\nError: Analysis crashed in a native dependency ({signal_name}).")
    if signal_number == signal.SIGILL:
        print(
            "This usually means OpenCV, PyTorch, or another binary package was built "
            "with CPU instructions this machine does not support."
        )
    print("Try running:")
    print("  python arlo_analyzer.py --self-test")
    print("If the self-test also crashes, reinstall compatible torch/ultralytics wheels.")
    print("If only this video crashes, re-encode or repair the MP4 before analysis.")


def run_isolated(args):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_worker, args=(result_queue, args))
    process.start()
    process.join()

    if process.exitcode and process.exitcode < 0:
        print_native_crash_help(process.exitcode)
        return 1

    if result_queue.empty():
        print("Error: Analysis stopped without returning a result.")
        return process.exitcode or 1

    result = result_queue.get()
    if not result["ok"]:
        if "error" in result:
            print(f"Error: {result['error']}")
        return int(result.get("exit_code") or 1)

    if args.self_test:
        return 0

    print_result(
        {
            "detected_objects": set(result["detected_objects"]),
            "object_counts": result["object_counts"],
            "object_confidences": result["object_confidences"],
            "annotated_paths": result["annotated_paths"],
            "annotated_dir": result["annotated_dir"],
        }
    )
    return 0


def main():
    args = parse_args()

    if not args.self_test and not args.video_path:
        print("Error: You must provide an MP4 file.")
        print("Usage: python arlo_analyzer.py <filename.mp4>")
        return 2

    if not args.no_isolation:
        return run_isolated(args)

    if args.self_test:
        self_test()
        return 0

    analysis_result = analyze_video(args)
    print_result(analysis_result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAnalysis interrupted.")
        sys.exit(130)
