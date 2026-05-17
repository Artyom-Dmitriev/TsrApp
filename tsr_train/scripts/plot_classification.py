import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("runs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -------- 1. Confusion matrix (row-normalised) --------
cm = np.load("runs/confusion_matrix.npy")
cm_norm = cm / cm.sum(axis=1, keepdims=True)
n_classes = cm.shape[0]

fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Доля предсказаний", fontsize=12)
cbar.ax.tick_params(labelsize=10)

ticks = list(range(0, n_classes, 5))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(ticks, fontsize=10)
ax.set_yticklabels(ticks, fontsize=10)

ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.3)
ax.tick_params(which="minor", length=0)

ax.set_xlabel("Предсказанный класс", fontsize=12)
ax.set_ylabel("Истинный класс", fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png")
plt.close(fig)


# -------- 2. Per-class F1 --------
with open("runs/test_report.json", encoding="utf-8") as f:
    payload = json.load(f)
report = payload["report"]

class_ids = list(range(n_classes))
f1s = np.array([report[str(c)]["f1-score"] for c in class_ids])
macro_f1 = report["macro avg"]["f1-score"]

threshold = 0.95
colors = ["#4682b4" if v >= threshold else "#e67e22" for v in f1s]

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
ax.bar(class_ids, f1s, color=colors, edgecolor="none", width=0.8)

ax.axhline(
    macro_f1,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"Macro-F1 = {macro_f1:.3f}",
)

y_min = max(0.0, min(f1s.min() - 0.02, 0.9))
ax.set_ylim(y_min, 1.005)
ax.set_xlim(-0.7, n_classes - 0.3)

x_ticks = list(range(0, n_classes, 2))
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, fontsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.set_xlabel("Номер класса", fontsize=12)
ax.set_ylabel("F1-мера", fontsize=12)
ax.legend(fontsize=11, loc="lower right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", color="lightgray", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIG_DIR / "per_class_f1.png")
plt.close(fig)


# -------- Output summary --------
print("Created:")
for name in ["confusion_matrix.png", "per_class_f1.png"]:
    p = FIG_DIR / name
    print(f"  {p}  {p.stat().st_size / 1024:.1f} KB")

worst = sorted(zip(class_ids, f1s), key=lambda t: t[1])[:5]
print("\nTop-5 lowest F1 (class, F1):")
for c, v in worst:
    print(f"  class {c:>2}  F1 = {v:.4f}")
