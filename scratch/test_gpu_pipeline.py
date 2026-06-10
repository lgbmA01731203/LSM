import os
import sys
import numpy as np
import cv2
import time

# Add parent directory to sys.path to find project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_hand_detector import GPUHandDetector

def test_pipeline():
    print("=== Testing GPU Hand Detector ===")
    
    # 1. Initialize detector
    detector = GPUHandDetector(
        palm_model_path="palm_detection.onnx",
        hand_model_path="handpose_estimation_mediapipe_2023feb.onnx"
    )
    
    print("Loading models...")
    success = detector.load()
    if not success:
        print("[FAIL] Failed to load detector models.")
        return
    print("[OK] Models loaded successfully.")
    
    # 2. Run on dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    print("Running detection on dummy black image...")
    
    start = time.time()
    result = detector.detect(dummy_img)
    end = time.time()
    
    print(f"[OK] Detection ran in {(end - start) * 1000:.2f} ms.")
    print(f"Number of hands detected: {len(result.hand_landmarks)}")
    
    # 3. Print execution providers
    import onnxruntime as ort
    print(f"Available ORT providers: {ort.get_available_providers()}")
    print("Verification completed.")

if __name__ == "__main__":
    test_pipeline()
