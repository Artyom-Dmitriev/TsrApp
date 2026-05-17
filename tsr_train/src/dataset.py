from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import GTSRB

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 43

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TransformedSubset(Dataset):
    """Wraps a base dataset + index list and applies its own transform.

    Why: torch.utils.data.Subset shares the underlying dataset's transform.
    Mutating base.transform to switch between train_tf and eval_tf would
    flip transforms globally for every subset that points at the same base.
    """

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base._samples[self.indices[i]]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_loaders(root="data", batch_size=32, val_split=0.2, seed=42, num_workers=2):
    full_train = GTSRB(root=root, split="train", download=False, transform=None)
    test_set = GTSRB(root=root, split="test", download=False, transform=eval_tf)

    labels = np.array([label for _, label in full_train._samples])
    indices = np.arange(len(labels))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels))

    train_subset = TransformedSubset(full_train, train_idx, train_tf)
    val_subset = TransformedSubset(full_train, val_idx, eval_tf)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _class_shares(labels_subset, num_classes=NUM_CLASSES):
    n = len(labels_subset)
    counts = Counter(int(l) for l in labels_subset)
    return np.array([counts.get(c, 0) / n for c in range(num_classes)])


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"Train: {n_train}")
    print(f"Val:   {n_val}")
    print(f"Test:  {n_test}")
    print(f"Train + Val: {n_train + n_val}")

    imgs, lbls = next(iter(train_loader))
    print(f"Batch images: {imgs.shape}")
    print(f"Batch labels: {lbls.shape}")

    full_train = train_loader.dataset.base
    all_labels = np.array([label for _, label in full_train._samples])
    train_labels = all_labels[train_loader.dataset.indices]
    val_labels = all_labels[val_loader.dataset.indices]

    gaps = np.abs(_class_shares(train_labels) - _class_shares(val_labels))
    print(f"Train<->Val class share gap: max={gaps.max():.5f}  mean={gaps.mean():.5f}")
    print(f"Train<->Val class share gap: max={gaps.max():.7f}  mean={gaps.mean():.7f}")
