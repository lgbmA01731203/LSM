import os
import sys
import numpy as np
import cv2
import time

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpu_hand_detector import GPUHandDetector

def benchmark():
    detector = GPUHandDetector(
        palm_model_path="palm_detection.onnx",
        hand_model_path="handpose_estimation_mediapipe_2023feb.onnx"
    )
    if not detector.load():
        print("Failed to load detector.")
        return
        
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        detector.detect(dummy_img)
        
    # Benchmark
    times = []
    for _ in range(50):
        t0 = time.time()
        detector.detect(dummy_img)
        times.append(time.time() - t0)
        
    avg_ms = np.mean(times) * 1000
    print(f"Average CPU Inference Time: {avg_ms:.2f} ms ({1000/avg_ms:.2f} FPS)")

if __name__ == "__main__":
    benchmark()
