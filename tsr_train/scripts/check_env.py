import sys

print(f"Python version: {sys.version}")

import torch
print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

import onnx
import onnxruntime
print(f"onnx version: {onnx.__version__}")
print(f"onnxruntime version: {onnxruntime.__version__}")
print("onnx and onnxruntime imported successfully")
