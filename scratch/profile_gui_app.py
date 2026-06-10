import os
import sys
import time
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import LSMApp
import tkinter as tk

def profile_gui():
    print("=== Profiling LSM GUI Application Update Loop ===")
    
    # Instantiate the application
    app = LSMApp()
    
    # Hide the window to avoid showing GUI during headless profiling
    app.withdraw()
    
    # Profile 100 updates of update_frame
    durations = []
    
    # Pre-populate pipeline with a dummy frame and features to simulate normal running
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_features = np.zeros(126)
    # Set fake latest frame in pipeline
    with app.pipeline._out_lock:
        app.pipeline._out_frame = dummy_frame
        app.pipeline._out_features = dummy_features
        
    print("Running 100 GUI updates...")
    for i in range(100):
        t_start = time.time()
        
        # We manually call update_frame contents (avoiding the after scheduling)
        # We can temporarily override self.after to be a no-op
        original_after = app.after
        app.after = lambda ms, func: None
        
        app.update_frame()
        
        app.after = original_after
        
        durations.append(time.time() - t_start)
        
    app.pipeline.stop()
    app.destroy()
    
    avg_duration = np.mean(durations)
    print(f"Average time per update_frame: {avg_duration * 1000:.2f} ms")
    print(f"Theoretical GUI max frame rate: {1.0 / avg_duration:.2f} FPS")
    
if __name__ == "__main__":
    profile_gui()
