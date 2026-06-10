import os
import sys
import numpy as np
import cv2
import time

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mp_handpose import MPHandPose

def benchmark():
    detector = MPHandPose(
        modelPath="handpose_estimation_mediapipe_2023feb.onnx"
    )
    
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_palm = np.zeros(19) # fake palm detection output
    dummy_palm[0:4] = [100, 100, 300, 300] # bbox
    # fake landmarks (7 landmarks * 2 coords = 14) spread out
    dummy_palm[4:18] = [150, 150, 160, 170, 170, 180, 180, 190, 190, 200, 200, 210, 210, 220]
    
    # Warmup
    for _ in range(10):
        detector.infer(dummy_img, dummy_palm)
        
    # Benchmark
    times = []
    for _ in range(50):
        t0 = time.time()
        detector.infer(dummy_img, dummy_palm)
        times.append(time.time() - t0)
        
    avg_ms = np.mean(times) * 1000
    print(f"Average Hand Pose CPU Inference Time: {avg_ms:.2f} ms ({1000/avg_ms:.2f} FPS)")

if __name__ == "__main__":
    benchmark()
