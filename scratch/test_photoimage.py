import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2

def test_photo_image():
    root = ctk.CTk()
    root.geometry("400x400")
    
    # Create dummy image
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), -1)
    
    img_pil = Image.fromarray(frame)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    label = ctk.CTkLabel(root, text="")
    label.pack(pady=20)
    
    # Configure with ImageTk.PhotoImage
    try:
        label.configure(image=img_tk)
        print("[OK] CTkLabel configured with ImageTk.PhotoImage successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to configure CTkLabel with ImageTk.PhotoImage: {e}")
        
    root.after(1000, root.destroy)
    root.mainloop()

if __name__ == "__main__":
    test_photo_image()
