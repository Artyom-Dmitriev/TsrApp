import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.dataset import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    model = build_model()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"model.fc.weight shape: {model.fc.weight.shape}")
