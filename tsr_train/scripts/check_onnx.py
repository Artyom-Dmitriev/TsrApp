import sys

import numpy as np
import onnxruntime as ort
import torch

from src.model import build_model

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ONNX_PATH = "exports/model.onnx"
CKPT_PATH = "runs/best.pt"


def _format_topk(logits, k=3):
    arr = np.asarray(logits).ravel()
    idx = np.argsort(arr)[::-1][:k]
    return "  ".join(f"{int(c)}={arr[c]:+.4f}" for c in idx)


def main():
    model = build_model()
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y_torch = model(x).numpy()
    y_onnx = sess.run(["logits"], {"input": x.numpy()})[0]

    diff = np.abs(y_torch - y_onnx)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Output shape (torch): {y_torch.shape}")
    print(f"Output shape (onnx):  {y_onnx.shape}")
    print(f"Max  |diff|: {max_diff:.3e}")
    print(f"Mean |diff|: {mean_diff:.3e}")
    print(f"Argmax (torch): {int(y_torch.argmax())}")
    print(f"Argmax (onnx):  {int(y_onnx.argmax())}")
    print(f"Top-3 (torch): {_format_topk(y_torch)}")
    print(f"Top-3 (onnx):  {_format_topk(y_onnx)}")

    if max_diff < 1e-4:
        print("✓ Sanity-check PASSED")
    else:
        print("✗ Sanity-check FAILED")
        return

    print("\nRunning 10 random inputs...")
    overall_max_diff = max_diff
    all_argmax_match = True
    for seed in range(10):
        torch.manual_seed(seed)
        xi = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            yt = model(xi).numpy()
        yo = sess.run(["logits"], {"input": xi.numpy()})[0]
        d = np.abs(yt - yo).max()
        argmax_match = int(yt.argmax()) == int(yo.argmax())
        all_argmax_match &= argmax_match
        overall_max_diff = max(overall_max_diff, d)
        print(
            f"  seed={seed}  max|diff|={d:.3e}  "
            f"argmax torch={int(yt.argmax())} onnx={int(yo.argmax())}  "
            f"{'OK' if argmax_match else 'MISMATCH'}"
        )

    print(f"\nOverall max |diff| across 10 runs: {overall_max_diff:.3e}")
    print(f"All argmax match: {all_argmax_match}")


if __name__ == "__main__":
    main()
