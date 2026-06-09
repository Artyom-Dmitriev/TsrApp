"""Export the trained YOLOv8n detector to detector.onnx.

Exports without built-in NMS (nms=False), so the ONNX output is the raw
prediction tensor of shape [1, 5, 8400] (cx, cy, w, h, conf over 8400 anchors).
Decoding + NMS happens in the consumer (check_yolo_onnx.py / the C# app).
"""

import shutil
import sys
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO

DETECTOR_DIR = Path(__file__).resolve().parent
DEFAULT_BEST = DETECTOR_DIR / "runs" / "train" / "weights" / "best.pt"
EXPORTS_DIR = DETECTOR_DIR / "exports"
OUTPUT_ONNX = EXPORTS_DIR / "detector.onnx"


def main():
    best_pt = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_BEST
    if not best_pt.exists():
        raise FileNotFoundError(
            f"{best_pt} not found -- run train_yolo.py first or pass the path to best.pt."
        )

    model = YOLO(str(best_pt))
    exported = model.export(
        format="onnx",
        opset=12,
        imgsz=640,
        dynamic=False,
        simplify=True,
        nms=False,
    )

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    exported_path = Path(exported)
    if exported_path.resolve() != OUTPUT_ONNX.resolve():
        shutil.copyfile(exported_path, OUTPUT_ONNX)
    print(f"\nExported: {OUTPUT_ONNX}")

    # Report the output tensor shape (expecting [1, 5, 8400]).
    sess = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
    for out in sess.get_outputs():
        print(f"output '{out.name}': shape {out.shape}, type {out.type}")


if __name__ == "__main__":
    main()
