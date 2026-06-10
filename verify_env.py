import sys
import os

print("=== LSM Environment Verification ===")
print(f"Python interpreter: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

try:
    import cv2
    print(f"[OK] OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"[FAIL] OpenCV: {e}")

try:
    import mediapipe as mp
    print(f"[OK] MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"[FAIL] MediaPipe: {e}")

try:
    import customtkinter as ctk
    print(f"[OK] CustomTkinter version: {ctk.__version__}")
except ImportError as e:
    print(f"[FAIL] CustomTkinter: {e}")

try:
    import sklearn
    print(f"[OK] Scikit-Learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"[FAIL] Scikit-Learn: {e}")

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     GPU Name: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"[FAIL] PyTorch: {e}")

try:
    import PIL
    print(f"[OK] Pillow version: {PIL.__version__}")
except ImportError as e:
    print(f"[FAIL] Pillow: {e}")

print("====================================")
