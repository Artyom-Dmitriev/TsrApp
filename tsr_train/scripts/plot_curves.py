import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIG_DIR = Path("runs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open("runs/history.json", encoding="utf-8") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
tr_loss = [h["tr_loss"] for h in history]
va_loss = [h["va_loss"] for h in history]
tr_acc = [h["tr_acc"] for h in history]
va_acc = [h["va_acc"] for h in history]
lrs = [h["lr"] for h in history]

TRAIN_KW = dict(color="#1f4e9e", linestyle="-", linewidth=2.0)
VAL_KW = dict(color="#e67e22", linestyle="--", linewidth=2.0)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="lightgray", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=10)


# 1. Loss curves
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.plot(epochs, tr_loss, label="Обучающая выборка", **TRAIN_KW)
ax.plot(epochs, va_loss, label="Валидационная выборка", **VAL_KW)
ax.set_xlabel("Эпоха", fontsize=12)
ax.set_ylabel("Значение функции потерь", fontsize=12)
ax.legend(fontsize=11)
_style(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "loss_curves.png")
plt.close(fig)

# 2. Accuracy curves
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.plot(epochs, tr_acc, label="Обучающая выборка", **TRAIN_KW)
ax.plot(epochs, va_acc, label="Валидационная выборка", **VAL_KW)
ax.set_xlabel("Эпоха", fontsize=12)
ax.set_ylabel("Точность", fontsize=12)
all_acc = tr_acc + va_acc
if min(all_acc) > 0.9:
    ax.set_ylim(min(all_acc) - 0.01, 1.005)
ax.legend(fontsize=11)
_style(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "accuracy_curves.png")
plt.close(fig)

# 3. LR schedule (log y)
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.plot(epochs, lrs, color="#2e8b57", linestyle="-", linewidth=2.0)
ax.set_yscale("log")
ax.set_xlabel("Эпоха", fontsize=12)
ax.set_ylabel("Скорость обучения (логарифм. шкала)", fontsize=12)
_style(ax)
plt.tight_layout()
plt.savefig(FIG_DIR / "lr_schedule.png")
plt.close(fig)

print("Created:")
for name in ["loss_curves.png", "accuracy_curves.png", "lr_schedule.png"]:
    p = FIG_DIR / name
    print(f"  {p}  {p.stat().st_size / 1024:.1f} KB")
