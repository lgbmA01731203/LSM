import os
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import customtkinter as ctk

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def benchmark():
    root = ctk.CTk()
    root.withdraw()
    
    # 1. Setup a standard tk.Label
    tk_frame = tk.Frame(root, width=800, height=600)
    tk_frame.pack_propagate(False)
    tk_label = tk.Label(tk_frame, width=800, height=600)
    tk_label.pack()
    
    # 2. Setup a ctk.CTkLabel
    ctk_frame = ctk.CTkFrame(root, width=800, height=600)
    ctk_label = ctk.CTkLabel(ctk_frame, text="")
    ctk_label.pack()
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (300, 300), (255, 255, 255), -1)
    
    print("=== Benchmarking Tkinter Label vs CTkLabel ===")
    
    # Measure: OpenCV resize + PIL conversion + standard tk.Label configure
    t_start = time.time()
    for _ in range(100):
        # OpenCV resize
        resized = cv2.resize(dummy_frame, (800, 600), interpolation=cv2.INTER_LINEAR)
        # PIL convert
        img_pil = Image.fromarray(resized)
        # PhotoImage
        img_tk = ImageTk.PhotoImage(image=img_pil)
        # tk.Label configure
        tk_label.configure(image=img_tk)
        tk_label.image = img_tk
    t_end = time.time()
    print(f"OpenCV resize + standard tk.Label: {(t_end - t_start)*1000/100:.2f} ms/frame")
    
    # Measure: PIL resize + standard tk.Label configure
    t_start = time.time()
    for _ in range(100):
        img_pil = Image.fromarray(dummy_frame)
        resized_pil = img_pil.resize((800, 600), Image.Resampling.BILINEAR)
        img_tk = ImageTk.PhotoImage(image=resized_pil)
        tk_label.configure(image=img_tk)
        tk_label.image = img_tk
    t_end = time.time()
    print(f"PIL resize + standard tk.Label: {(t_end - t_start)*1000/100:.2f} ms/frame")
    
    # Measure: ctk.CTkImage + CTkLabel configure
    t_start = time.time()
    for _ in range(100):
        img_pil = Image.fromarray(dummy_frame)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(800, 600))
        ctk_label.configure(image=img_tk)
        ctk_label.image = img_tk
    t_end = time.time()
    print(f"ctk.CTkImage + CTkLabel configure: {(t_end - t_start)*1000/100:.2f} ms/frame")
    
    root.destroy()

if __name__ == "__main__":
    benchmark()
