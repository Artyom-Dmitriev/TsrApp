"""Fine-tune yolov8n.pt on the GTSDB single-class detector dataset.

Run prepare_gtsdb.py first so that detector/data/data.yaml and the
images/labels splits exist.
"""

from pathlib import Path

from ultralytics import YOLO

DETECTOR_DIR = Path(__file__).resolve().parent
DATA_YAML = (DETECTOR_DIR / "data" / "data.yaml").resolve()
RUNS_DIR = DETECTOR_DIR / "runs"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found -- run prepare_gtsdb.py first."
        )

    model = YOLO("yolov8n.pt")

    # Pass data= as an absolute path so ultralytics resolves the dataset here
    # and not under its global settings datasets_dir.
    results = model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=100,
        batch=16,
        patience=20,
        project=str(RUNS_DIR),
        name="train",
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    # results.box.map50 = mAP@0.5 from the final validation pass.
    try:
        map50 = results.box.map50
    except AttributeError:
        map50 = None

    print(f"\nBest weights: {best_pt}")
    if map50 is not None:
        print(f"mAP@0.5: {map50:.4f}")
    else:
        print("mAP@0.5: see validation summary above")


if __name__ == "__main__":
    main()
