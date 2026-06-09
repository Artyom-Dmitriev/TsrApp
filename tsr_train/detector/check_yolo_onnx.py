"""Sanity-check detector.onnx: run one image through onnxruntime, decode the
raw [1, 5, 8400] output, draw the detected boxes, and save the result for a
visual look.

Usage:
    python check_yolo_onnx.py [image_path]

If no image is given, the first frame in data/images/val/ is used.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

DETECTOR_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = DETECTOR_DIR / "exports" / "detector.onnx"
VAL_IMAGES = DETECTOR_DIR / "data" / "images" / "val"
OUTPUT_IMG = DETECTOR_DIR / "exports" / "check_result.png"

IMGSZ = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def letterbox(img, size=IMGSZ, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a square `size`x`size` canvas.

    Returns (padded_image, scale, pad_x, pad_y) so boxes can be mapped back.
    """
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess(img):
    canvas, scale, pad_x, pad_y = letterbox(img)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # HWC -> NCHW, add batch
    return np.ascontiguousarray(arr), scale, pad_x, pad_y


def iou(box, boxes):
    """IoU of one xyxy box against an array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    return keep


def decode(output, scale, pad_x, pad_y):
    """Decode raw YOLOv8 output [1, 5, 8400] -> (xyxy boxes, scores) in the
    coordinate space of the original (pre-letterbox) image."""
    preds = output[0].T  # [8400, 5]: cx, cy, w, h, conf
    conf = preds[:, 4]
    mask = conf >= CONF_THRESHOLD
    preds = preds[mask]
    conf = conf[mask]
    if preds.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,))

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    # Undo letterbox: subtract padding, divide by scale.
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, conf)
    return boxes[keep], conf[keep]


def main():
    if not DEFAULT_ONNX.exists():
        raise FileNotFoundError(
            f"{DEFAULT_ONNX} not found -- run export_yolo_onnx.py first."
        )

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        candidates = sorted(VAL_IMAGES.glob("*.png"))
        if not candidates:
            raise FileNotFoundError(
                f"no image given and none found in {VAL_IMAGES}"
            )
        image_path = candidates[0]

    img = Image.open(image_path).convert("RGB")
    inp, scale, pad_x, pad_y = preprocess(img)

    sess = ort.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: inp})[0]

    boxes, scores = decode(output, scale, pad_x, pad_y)

    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), f"{s:.2f}", fill=(255, 0, 0))

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_IMG)
    print(f"Image: {image_path}")
    print(f"Detections: {len(boxes)}")
    print(f"Saved: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
