import sys
import os
import time

# Add current dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera_pipeline import LSMCameraPipeline

def test_pipeline_run():
    print("=== Testing Camera Pipeline ===")
    pipeline = LSMCameraPipeline()
    
    print("Starting pipeline...")
    pipeline.start()
    
    print("Running for 5 seconds...")
    for i in range(5):
        time.sleep(1.0)
        frame, features = pipeline.get_latest()
        if frame is not None:
            print(f"[{i+1}s] Frame shape: {frame.shape}, Features non-zero elements: {np.count_nonzero(features)}")
        else:
            print(f"[{i+1}s] Waiting for frame...")
            
    print("Stopping pipeline...")
    pipeline.stop()
    print("Test finished successfully!")

if __name__ == "__main__":
    import numpy as np
    test_pipeline_run()
