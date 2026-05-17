"""Pick 20 random GTSRB test images, convert them to PNG, drop them next to
the TsrApp project so they can be fed into the WPF UI without each user
having to figure out PIL / file dialogs themselves.

Deterministic via seed=42 so re-running gives the same set.

Source PPMs:  data/gtsrb/GTSRB/Final_Test/Images/*.ppm
Labels CSV:   data/gtsrb/GT-final_test.csv  (semicolon-separated, ClassId col)
Output:       ../TsrApp/sample_images/test_NN.png
"""

import csv
import random
import sys
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC_DIR = Path("data/gtsrb/GTSRB/Final_Test/Images")
GT_CSV = Path("data/gtsrb/GT-final_test.csv")
OUT_DIR = Path("../TsrApp/sample_images")
N = 20
SEED = 42

GTSRB_NAMES = [
    "Speed limit (20 km/h)", "Speed limit (30 km/h)", "Speed limit (50 km/h)",
    "Speed limit (60 km/h)", "Speed limit (70 km/h)", "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)", "Speed limit (100 km/h)",
    "Speed limit (120 km/h)", "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons",
]


def load_labels() -> dict[str, int]:
    labels: dict[str, int] = {}
    with GT_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            labels[row["Filename"]] = int(row["ClassId"])
    return labels


def main():
    labels = load_labels()
    all_ppms = sorted(p.name for p in SRC_DIR.glob("*.ppm"))
    if len(all_ppms) < N:
        raise SystemExit(f"only {len(all_ppms)} ppm files under {SRC_DIR}, need {N}")

    rng = random.Random(SEED)
    chosen = rng.sample(all_ppms, N)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"output dir: {OUT_DIR.resolve()}")
    print(f"{'#':>2}  {'src':<14}  {'out':<14}  {'class':>3}  name")
    print("-" * 90)

    for i, fname in enumerate(chosen):
        cls = labels[fname]
        out_name = f"test_{i:02d}.png"
        out_path = OUT_DIR / out_name
        with Image.open(SRC_DIR / fname) as img:
            img.convert("RGB").save(out_path)
        print(f"{i:>2}  {fname:<14}  {out_name:<14}  {cls:>3}  {GTSRB_NAMES[cls]}")


if __name__ == "__main__":
    main()
