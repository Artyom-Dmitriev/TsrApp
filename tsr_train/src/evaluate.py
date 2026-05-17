import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_loaders
from src.model import build_model


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: {DEVICE}")

    _, _, test_loader = get_loaders(batch_size=64)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("runs/best.pt", map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            preds = logits.argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_acc = (y_true == y_pred).mean()

    report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
    report_str = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    np.save("runs/y_true.npy", y_true)
    np.save("runs/y_pred.npy", y_pred)
    np.save("runs/confusion_matrix.npy", cm)
    Path("runs/test_report.txt").write_text(
        f"Test accuracy: {test_acc:.4f}\n\n" + report_str
    )
    Path("runs/test_report.json").write_text(
        json.dumps(
            {"test_accuracy": float(test_acc), "report": report_dict},
            indent=2,
        )
    )

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro-F1:      {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted-F1:   {report_dict['weighted avg']['f1-score']:.4f}")
    print()
    print(report_str)


if __name__ == "__main__":
    main()
