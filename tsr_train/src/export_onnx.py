import json
from pathlib import Path

import torch

from src.model import build_model

GTSRB_NAMES = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

assert len(GTSRB_NAMES) == 43, f"expected 43 names, got {len(GTSRB_NAMES)}"


def main():
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    model = build_model()
    model.load_state_dict(torch.load("runs/best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = export_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    labels_path = export_dir / "labels.json"
    labels_path.write_text(
        json.dumps(
            {str(i): n for i, n in enumerate(GTSRB_NAMES)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for p in (onnx_path, labels_path):
        size_kb = p.stat().st_size / 1024
        if size_kb >= 1024:
            print(f"{p}  {size_kb / 1024:.2f} MB")
        else:
            print(f"{p}  {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
