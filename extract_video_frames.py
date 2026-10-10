import argparse
import re
import sys
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced JPEG frames from one MP4 file or every new MP4 in a folder."
    )
    parser.add_argument("path", help="Path to an MP4 file or a folder containing MP4 files.")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of JPEG frames to extract. Default: 5.",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_frames",
        help="Folder where JPEG frames are saved. Default: extracted_frames.",
    )
    return parser.parse_args()


def safe_filename(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def get_frame_positions(total_frames, count):
    if count <= 0:
        print("Error: --count must be greater than 0.")
        raise SystemExit(1)

    if total_frames <= 0:
        print("Error: Could not determine the number of frames in the video.")
        raise SystemExit(1)

    if count == 1:
        return [total_frames // 2]

    last_frame = total_frames - 1
    return [
        round(index * last_frame / (count - 1))
        for index in range(count)
    ]


def mark_video_as_done(video_file):
    if video_file.name.startswith("ok_"):
        return video_file

    done_path = video_file.with_name(f"ok_{video_file.name}")
    if done_path.exists():
        print(f"Warning: Could not rename '{video_file}' because '{done_path}' already exists.")
        return video_file

    video_file.rename(done_path)
    print(f"Renamed MP4 to '{done_path}'.")
    return done_path


def find_videos(path):
    if not path.exists():
        print(f"Error: Path '{path}' was not found.")
        raise SystemExit(1)

    if path.is_file():
        if path.suffix.lower() != ".mp4":
            print(f"Error: '{path}' is not an MP4 file.")
            raise SystemExit(1)
        if path.name.startswith("ok_"):
            print(f"Skipping '{path}'; filename already starts with ok_.")
            return []
        return [path]

    videos = [
        video_path
        for video_path in sorted(path.glob("*.mp4"))
        if not video_path.name.startswith("ok_")
    ]
    return videos


def extract_frames_from_video(video_file, output_dir, count):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Warning: Could not open video file '{video_file}', skipping.")
        return 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_key = safe_filename(video_file.stem)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    positions = get_frame_positions(total_frames, count)
    saved_count = 0

    try:
        for image_number, frame_number in enumerate(positions, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_number}.")
                continue

            timestamp_seconds = frame_number / fps if fps else 0
            filename = (
                f"{video_key}_frame-{frame_number:06d}"
                f"_time-{timestamp_seconds:07.2f}_img-{image_number:02d}.jpg"
            )
            if cv2.imwrite(str(output_path / filename), frame):
                saved_count += 1
    finally:
        cap.release()

    print(
        f"Saved {saved_count} JPEG frame(s) from '{video_file}' "
        f"to '{output_path}'."
    )

    if saved_count:
        mark_video_as_done(video_file)

    return saved_count


def extract_frames(path, output_dir, count):
    videos = find_videos(Path(path))
    if not videos:
        print(f"No unprocessed MP4 files found in '{path}'.")
        return 1

    total_saved = 0
    processed_count = 0
    for video_file in videos:
        print(f"Extracting frames from '{video_file}'...")
        saved_count = extract_frames_from_video(video_file, output_dir, count)
        if saved_count:
            processed_count += 1
            total_saved += saved_count

    print(
        f"\nDone. Saved {total_saved} JPEG frame(s) from {processed_count} video(s). "
        f"Skipped files starting with ok_."
    )
    return 0 if processed_count else 1


def main():
    args = parse_args()
    return extract_frames(args.path, args.output_dir, args.count)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nFrame extraction interrupted.")
        sys.exit(130)
