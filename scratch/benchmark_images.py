import customtkinter as ctk
import time
from PIL import Image, ImageTk
import numpy as np
import cv2

def benchmark():
    print("=== CustomTkinter CTkImage Real Rendering Benchmark ===")
    
    import tkinter as tk
    root = tk.Tk()  # Needs a tk context for PhotoImage operations
    
    # Large frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    target_size = (800, 600)
    
    print("\n--- Benchmarking actual PhotoImage creation (100 iterations) ---")
    
    # Method A: Instantiating CTkImage every frame + getting photo image
    t0 = time.time()
    for _ in range(100):
        img_pil = Image.fromarray(frame)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=target_size)
        _ = img_tk.create_scaled_photo_image(1.0, "dark")
    print(f"A. CTkImage instantiation + create_scaled_photo_image (1.0 scale): {(time.time() - t0)*1000:.2f} ms")
    
    # Method B: cv2.resize (LINEAR) + Instantiating CTkImage every frame
    t0 = time.time()
    for _ in range(100):
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(resized)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=target_size)
        _ = img_tk.create_scaled_photo_image(1.0, "dark")
    print(f"B. cv2.resize (LINEAR) + CTkImage instantiation: {(time.time() - t0)*1000:.2f} ms")

    # Method C: cv2.resize (NEAREST) + Instantiating CTkImage every frame
    t0 = time.time()
    for _ in range(100):
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
        img_pil = Image.fromarray(resized)
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=target_size)
        _ = img_tk.create_scaled_photo_image(1.0, "dark")
    print(f"C. cv2.resize (NEAREST) + CTkImage instantiation: {(time.time() - t0)*1000:.2f} ms")

    # Method D: Standard ImageTk.PhotoImage with cv2.resize (LINEAR) - No CTkImage
    t0 = time.time()
    for _ in range(100):
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(resized)
        _ = ImageTk.PhotoImage(image=img_pil)
    print(f"D. ImageTk.PhotoImage + cv2.resize (LINEAR): {(time.time() - t0)*1000:.2f} ms")

    # Method E: Standard ImageTk.PhotoImage with cv2.resize (NEAREST) - No CTkImage
    t0 = time.time()
    for _ in range(100):
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
        img_pil = Image.fromarray(resized)
        _ = ImageTk.PhotoImage(image=img_pil)
    print(f"E. ImageTk.PhotoImage + cv2.resize (NEAREST): {(time.time() - t0)*1000:.2f} ms")

    # Method F: ImageTk.PhotoImage with no resizing (640x480)
    t0 = time.time()
    for _ in range(100):
        img_pil = Image.fromarray(frame)
        _ = ImageTk.PhotoImage(image=img_pil)
    print(f"F. ImageTk.PhotoImage (No resizing, 640x480): {(time.time() - t0)*1000:.2f} ms")

    root.destroy()

if __name__ == "__main__":
    benchmark()
