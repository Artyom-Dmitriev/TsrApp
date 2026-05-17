"""Generate reference input/logits/probs from the Python pipeline so that the
C# implementation can be checked for parity.

Picks the first test image from GTSRB, applies the exact eval_tf used during
training and evaluation, runs the exported ONNX model via onnxruntime, and
saves:
  ../TsrApp/Reference/test_image.png       — raw RGB image as written by PIL
  ../TsrApp/Reference/expected_input.bin   — float32, len 3*224*224, CHW
  ../TsrApp/Reference/expected_logits.bin  — float32, len 43
  ../TsrApp/Reference/expected_probs.bin   — float32, len 43, softmax(logits)
  ../TsrApp/Reference/post_resize_uint8.bin — uint8, 224*224*3 HWC (debug)
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.datasets import GTSRB

from src.dataset import eval_tf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ONNX_PATH = "exports/model.onnx"
REF_DIR = Path("../TsrApp/Reference")

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


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)

    test_set = GTSRB(root="data", split="test", download=False, transform=None)
    path, true_label = test_set._samples[0]
    print(f"Source image: {path}")
    print(f"True label:   {true_label} ({GTSRB_NAMES[true_label]})")

    pil_img = Image.open(path).convert("RGB")
    png_path = REF_DIR / "test_image.png"
    pil_img.save(png_path)
    print(f"Saved        {png_path}  ({pil_img.size[0]}x{pil_img.size[1]})")

    pre = np.array(pil_img)
    print(f"PRE-resize pixel(0,0)   R={pre[0,0,0]} G={pre[0,0,1]} B={pre[0,0,2]}")
    print(f"PRE-resize pixel(10,10) R={pre[10,10,0]} G={pre[10,10,1]} B={pre[10,10,2]}")
    resized = pil_img.resize((224, 224), Image.BILINEAR)
    post = np.array(resized)
    print(f"POST-resize pixel(0,0)   R={post[0,0,0]} G={post[0,0,1]} B={post[0,0,2]}")
    print(f"POST-resize pixel(112,112) R={post[112,112,0]} G={post[112,112,1]} B={post[112,112,2]}")
    (REF_DIR / "post_resize_uint8.bin").write_bytes(post.astype(np.uint8).tobytes())
    print(f"Saved        post_resize_uint8.bin  ({post.shape}, dtype={post.dtype})")

    tensor = eval_tf(pil_img)
    assert tensor.shape == (3, 224, 224), tensor.shape
    arr = tensor.numpy().astype(np.float32, copy=False).ravel()
    assert arr.size == 3 * 224 * 224
    (REF_DIR / "expected_input.bin").write_bytes(arr.tobytes())
    print(f"Saved        expected_input.bin   len={arr.size}  "
          f"min={arr.min():+.4f}  max={arr.max():+.4f}")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_input = arr.reshape(1, 3, 224, 224)
    logits = sess.run(["logits"], {"input": onnx_input})[0].ravel().astype(np.float32)
    assert logits.size == 43, logits.size
    (REF_DIR / "expected_logits.bin").write_bytes(logits.tobytes())

    exp = np.exp(logits - logits.max())
    probs = (exp / exp.sum()).astype(np.float32)
    (REF_DIR / "expected_probs.bin").write_bytes(probs.tobytes())

    top3_idx = np.argsort(probs)[::-1][:3]
    print()
    print(f"ONNX argmax: {int(logits.argmax())} ({GTSRB_NAMES[int(logits.argmax())]})")
    print("Top-3:")
    for c in top3_idx:
        print(f"  {int(c):2d} {GTSRB_NAMES[int(c)]:50s} "
              f"logit={logits[c]:+8.4f}  p={probs[c]:.6f}")

    match = int(logits.argmax()) == true_label
    print()
    print(f"Top-1 matches true label: {'YES' if match else 'NO'}")
    if not match:
        print("WARNING: model misclassifies the reference image. "
              "Parity check will still work but you'll be checking against "
              "a wrong-but-deterministic prediction.")


if __name__ == "__main__":
    main()
