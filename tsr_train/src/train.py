import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import get_loaders
from src.model import build_model

EPOCHS = 20
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    correct = 0
    n = 0

    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

    return total_loss / n, correct / n


def main():
    print(f"DEVICE: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    Path("runs").mkdir(exist_ok=True)

    train_loader, val_loader, _ = get_loaders(batch_size=BATCH)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    bad_epochs = 0
    t_start = time.time()
    last_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_loss)
        dt = time.time() - t0

        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "va_loss": va_loss,
            "va_acc": va_acc,
            "lr": current_lr,
            "time_sec": dt,
        })

        print(
            f"E{epoch:02d}  tr={tr_loss:.3f}/{tr_acc:.3f}  "
            f"va={va_loss:.3f}/{va_acc:.3f}  lr={current_lr:.1e}  ({dt:.0f}s)",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            bad_epochs = 0
            torch.save(model.state_dict(), "runs/best.pt")
            print("  -> new best, saved runs/best.pt", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                last_epoch = epoch
                break
        last_epoch = epoch

    Path("runs/history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2)
    )

    total_time = time.time() - t_start
    print(
        f"Done. Epochs ran: {last_epoch}  "
        f"best val_loss={best_val_loss:.4f}  best val_acc={best_val_acc:.4f}  "
        f"total time={total_time:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
