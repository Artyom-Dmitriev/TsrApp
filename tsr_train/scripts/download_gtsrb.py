from pathlib import Path

from torchvision.datasets import GTSRB

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Downloading GTSRB into: {DATA_ROOT}")

train_set = GTSRB(root=str(DATA_ROOT), split="train", download=True)
test_set = GTSRB(root=str(DATA_ROOT), split="test", download=True)

num_train = len(train_set)
num_test = len(test_set)
num_classes = len({label for _, label in train_set._samples})

print(f"Train images: {num_train}")
print(f"Test images:  {num_test}")
print(f"Unique classes in train: {num_classes}")
