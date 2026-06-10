import os
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageTk

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import LSMApp
import tkinter as tk

def profile_gui_breakdown():
    print("=== Profiling LSM GUI Application Update Loop Breakdown ===")
    
    app = LSMApp()
    app.withdraw()
    
    # Mock winfo_width and winfo_height
    app.camera_frame.winfo_width = lambda: 800
    app.camera_frame.winfo_height = lambda: 600
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (300, 300), (255, 255, 255), -1)
    
    dummy_features = np.zeros(126)
    dummy_features[10:20] = 0.5
    
    with app.pipeline._out_lock:
        app.pipeline._out_frame = dummy_frame
        app.pipeline._out_features = dummy_features
        
    t_get_latest = []
    t_inference = []
    t_pil_conv = []
    t_ctk_img = []
    t_configure = []
    
    for i in range(100):
        # Override self.after to be a no-op
        original_after = app.after
        app.after = lambda ms, func: None
        
        # Step 1: get_latest
        t0 = time.time()
        frame, features = app.pipeline.get_latest()
        t1 = time.time()
        t_get_latest.append(t1 - t0)
        
        # Step 2: inference
        t0 = time.time()
        active_mode = app.mode_var.get()
        if features is not None:
            new_token, raw_prediction, confidence = app.translator.process_frame(features, mode=active_mode)
        t1 = time.time()
        t_inference.append(t1 - t0)
        
        # Step 3: PIL conversion
        t0 = time.time()
        if frame is not None:
            img_pil = Image.fromarray(frame)
        t1 = time.time()
        t_pil_conv.append(t1 - t0)
        
        # Step 4: CTkImage instantiation
        t0 = time.time()
        if frame is not None:
            w = app.camera_frame.winfo_width()
            h = app.camera_frame.winfo_height()
            import customtkinter as ctk
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(w, h))
        t1 = time.time()
        t_ctk_img.append(t1 - t0)
        
        # Step 5: label configure
        t0 = time.time()
        if frame is not None:
            app.video_label.configure(image=img_tk, text="")
            app.video_label.image = img_tk
        t1 = time.time()
        t_configure.append(t1 - t0)
        
        app.after = original_after
        
    app.pipeline.stop()
    app.destroy()
    
    print(f"Average get_latest: {np.mean(t_get_latest)*1000:.2f} ms")
    print(f"Average inference: {np.mean(t_inference)*1000:.2f} ms")
    print(f"Average PIL conversion: {np.mean(t_pil_conv)*1000:.2f} ms")
    print(f"Average CTkImage creation: {np.mean(t_ctk_img)*1000:.2f} ms")
    print(f"Average Widget Configure: {np.mean(t_configure)*1000:.2f} ms")
    
if __name__ == "__main__":
    profile_gui_breakdown()
