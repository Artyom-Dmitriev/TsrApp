"""Prepare the GTSDB (FullIJCNN2013) dataset for a single-class YOLO detector.

Downloads the archive, converts the PPM frames to PNG, rewrites the GTSDB
bounding-box annotations (gt.txt) into YOLO txt files (one per image), and
splits the images 80/20 into train/val. All 43 GTSRB class ids collapse into a
single detector class 0 ("traffic_sign") -- this detector only finds *where*
the signs are; the existing ResNet-18 classifier decides *which* sign it is.

Layout produced (everything under detector/data/):

    data/raw/FullIJCNN2013/...        # downloaded + extracted source
    data/images/{train,val}/*.png     # converted frames
    data/labels/{train,val}/*.txt     # YOLO annotations
    data/data.yaml                    # ultralytics dataset descriptor
"""

import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

# GTSDB / FullIJCNN2013, hosted by the dataset authors on the ERDA archive.
GTSDB_URL = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"

DETECTOR_DIR = Path(__file__).resolve().parent
DATA_DIR = DETECTOR_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DIR / "FullIJCNN2013.zip"
SOURCE_DIR = RAW_DIR / "FullIJCNN2013"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
DATA_YAML = DATA_DIR / "data.yaml"

VAL_FRACTION = 0.20
SEED = 42


def download_and_extract():
    """Fetch and unzip the GTSDB archive, skipping work already done."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        if not ZIP_PATH.exists():
            print(f"Downloading GTSDB from {GTSDB_URL} ...")
            urllib.request.urlretrieve(GTSDB_URL, ZIP_PATH)
            print(f"  saved {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"Archive already present: {ZIP_PATH}")

        print("Extracting ...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
    else:
        print(f"Source already extracted: {SOURCE_DIR}")

    gt_path = SOURCE_DIR / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found at {gt_path}")
    return gt_path


def parse_annotations(gt_path):
    """Parse gt.txt into {ppm_filename: [(left, top, right, bottom), ...]}.

    gt.txt format (semicolon-separated, one box per line):
        00000.ppm;left;top;right;bottom;classId
    A frame may appear on several lines (multiple signs) or not at all
    (background-only frame).
    """
    boxes = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            name = parts[0]
            left, top, right, bottom = (int(float(p)) for p in parts[1:5])
            boxes[name].append((left, top, right, bottom))
    return boxes


def to_yolo_line(box, width, height):
    """Convert an (left, top, right, bottom) box to a normalized YOLO line."""
    left, top, right, bottom = box
    cx = (left + right) / 2.0 / width
    cy = (top + bottom) / 2.0 / height
    w = (right - left) / width
    h = (bottom - top) / height
    # Class 0 = traffic_sign (single-class detector).
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    gt_path = download_and_extract()
    annotations = parse_annotations(gt_path)

    # Every .ppm frame in the source dir is a sample (some have no signs).
    ppm_files = sorted(SOURCE_DIR.glob("*.ppm"))
    if not ppm_files:
        raise FileNotFoundError(f"no .ppm frames found in {SOURCE_DIR}")

    rng = random.Random(SEED)
    shuffled = ppm_files[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    val_set = set(shuffled[:n_val])

    for split in ("train", "val"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxes_total = 0
    for ppm in ppm_files:
        split = "val" if ppm in val_set else "train"
        stem = ppm.stem

        # PPM -> PNG (lossless); ultralytics reads PNG reliably.
        with Image.open(ppm) as img:
            img = img.convert("RGB")
            width, height = img.size
            img.save(IMAGES_DIR / split / f"{stem}.png")

        lines = [to_yolo_line(b, width, height) for b in annotations.get(ppm.name, [])]
        boxes_total += len(lines)
        (LABELS_DIR / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        counts[split] += 1

    # Absolute path in data.yaml so ultralytics does not look in its global
    # datasets_dir and fail with "dataset not found".
    data_root = DATA_DIR.resolve().as_posix()
    DATA_YAML.write_text(
        f"path: {data_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        'names: ["traffic_sign"]\n',
        encoding="utf-8",
    )

    print(f"\nFrames: {len(ppm_files)} total -> {counts['train']} train / {counts['val']} val")
    print(f"Bounding boxes: {boxes_total}")
    print(f"data.yaml written: {DATA_YAML}")


if __name__ == "__main__":
    main()
