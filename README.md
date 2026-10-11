# Arlo Video Analyzer

This project contains small Python tools for extracting frames from Arlo MP4 videos, creating YOLO labels, generating training material, and running object detection.

## Setup

Create a virtual environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For YOLO training, use the training requirements:

```bash
python -m pip install -r requirements-yolo-training.txt
```

## Scripts

### `video_analyzer.py`

Analyzes one MP4 file with a YOLO model. It reports detected object classes and can save one annotated image per detected class.

Example:

```bash
python video_analyzer.py /path/to/video.mp4 --model runs/detect/train-6/weights/best.pt --confidence 0.35 --min-object-hits 5 --imgsz 1280
```

Useful options:

```bash
--model yolov8s.pt
--confidence 0.45
--imgsz 960
--min-object-hits 5
--annotated-dir annotated_objects
```

### `annotate_raw_videos.py`

Processes MP4 files from a raw video folder and creates training/review material. For each detected frame it saves:

- the original JPEG frame
- an annotated JPEG frame with detection rectangles
- a YOLO `.txt` label file

By default it scans `raw/`, writes flat output files to `raw_analyzed/`, and skips videos listed in `raw_analyzed/analyzed_videos.txt`. Existing `.done` marker files are imported into that list automatically. If no matching detections are saved from a video, the MP4 is copied to `raw_second_check/`.

Run on all new MP4 files in `raw/`:

```bash
python annotate_raw_videos.py
```

Run on one specific MP4:

```bash
python annotate_raw_videos.py raw/example.mp4
```

Useful options:

```bash
--model yolov8s.pt
--second-model runs/detect/train-6/weights/best.pt
--confidence 0.35
--imgsz 960
--images-per-video 10
--classes cat,dog,robot_mower,person,car
```

### `extract_video_frames.py`

Extracts evenly spaced JPEG frames from one MP4 file or from every unprocessed MP4 in a folder. Files whose names already start with `ok_` are skipped. After frames are successfully extracted from a video, the MP4 is renamed with an `ok_` prefix.

Extract five frames from one MP4:

```bash
python extract_video_frames.py raw_second_check/example.mp4
```

Extract five frames from every unprocessed MP4 in a folder:

```bash
python extract_video_frames.py raw_second_check
```

Change count or output folder:

```bash
python extract_video_frames.py raw_second_check --count 5 --output-dir extracted_frames
```

### `label_image_rectangle.py`

Interactive manual labeling tool. It shows an image, lets you draw one rectangle, choose the class inside the image window, and saves a YOLO label file.

Label one image:

```bash
python label_image_rectangle.py extracted_frames/example.jpg
```

Label every JPEG in a folder that does not already have a matching `.txt` label file:

```bash
python label_image_rectangle.py extracted_frames
```

Default classes:

```text
0: cat
1: dog
2: robot_mower
3: person
4: car
```

Controls:

```text
drag mouse  draw rectangle
s / Enter   open class selector
0-9         choose class and save label
r           reset / go back
q / Esc     quit
```

### `change_label_class.py`

Changes the class ID in every YOLO `.txt` label file in a folder while keeping the bounding box coordinates unchanged.

Change every label in one folder to class `2`:

```bash
python change_label_class.py raw_analyzed/robot_mower 2
```

Include subfolders:

```bash
python change_label_class.py dataset/labels 2 --recursive
```

Preview changes without writing files:

```bash
python change_label_class.py raw_analyzed/robot_mower 2 --dry-run
```

### `prefix_labels_by_class.py`

Renames every YOLO `.txt` label file and matching JPEG in a folder to start with the class name from `dataset/data.yaml`.

Preview changes:

```bash
python prefix_labels_by_class.py extracted_frames/wrong_dogs --dry-run
```

Rename files:

```bash
python prefix_labels_by_class.py extracted_frames/wrong_dogs
```

### `move_val_labels_for_images.py`

Moves matching label files from `dataset/labels/train` to `dataset/labels/val` for every JPEG in `dataset/images/val`.

Preview changes:

```bash
python move_val_labels_for_images.py --dry-run
```

Move the labels:

```bash
python move_val_labels_for_images.py
```

## Training

YOLO datasets use matching image and label paths:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Each label file uses YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalized from `0` to `1`.
 
Example training command:

```bash
yolo detect train model=yolov8s.pt data=dataset/data.yaml epochs=50 imgsz=960 batch=1 workers=0 cache=False plots=False device=cpu
```

## Process for training

- Copy new mp4-files to raw 
- Run `python annotate_raw_videos.py  --model yolov8n.pt  --confidence 0.4`
- Check raw_analyzed for annotated jpg-files. Check content, and move all related files to a subfolder for that object class
- For mp4-files where no objects could be identified, these end up in raw_second_check, check these manually and move them into subfolders depending on content.
- Extract jpeg images from the subfolder above `python extract_video_frames.py raw_second_check/animals`
- For every extracted imaged, make a rectangle and choose class - ´python label_image_rectangle.py extracted_frames --class-id 2`
- Move jpegs and txt-files into data_set-folders
- Train... `yolo detect train model=yolov8s.pt data=dataset/data.yaml epochs=50 imgsz=960 batch=1 workers=0 cache=False plots=False device=cpu`
 
## Analyze video

- `python video_analyzer.py raw/AHB14578D20FA_000004fe_20260415_014820.mp4 --model runs/detect/train-4/weights/best.pt --confidence 0.35 --min-object-hits 5 --imgsz 960`
