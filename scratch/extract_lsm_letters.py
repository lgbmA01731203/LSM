import os
import sys
import shutil
import cv2
import numpy as np

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpu_hand_detector import GPUHandDetector
import utils
import train_models

DATASET_ROOT = r"C:\Users\capa_\OneDrive\Desktop\LSM\archive (3)\MSL-ABC"
OUTPUT_ROOT = os.path.join("data", "static_letters")
CLASS_MAP = {
    '1': 'a',
    '2': 'b',
    '3': 'c',
    '4': 'd',
    '5': 'e'
}

def clean_output_directories():
    """Deletes and recreates the target folders so they have no old or mixed data."""
    for letter in CLASS_MAP.values():
        dest_dir = os.path.join(OUTPUT_ROOT, letter)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
    # Also ensure 'llorar' is cleaned up
    llorar_dir = os.path.join(OUTPUT_ROOT, 'llorar')
    if os.path.exists(llorar_dir):
        shutil.rmtree(llorar_dir)

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"[ERROR] Dataset root not found: {DATASET_ROOT}")
        return
        
    print(f"Starting keypoint extraction from {DATASET_ROOT}...")
    clean_output_directories()
    
    # Initialize GPU Hand Detector
    detector = GPUHandDetector(palm_model_path="palm_detection.onnx", hand_model_path="handpose_estimation_mediapipe_2023feb.onnx")
    if not detector.load():
        print("[ERROR] Failed to load GPUHandDetector.")
        return
        
    # Automatically detect all lsm-abc-* subdirectories
    subdirs = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("lsm-abc-")]
    print(f"Found letter directories in archive (3): {subdirs}")
    
    step = 90
    counts = {letter: 0 for letter in CLASS_MAP.values()}
    skipped_count = 0
    total_processed = 0
    
    for folder in subdirs:
        folder_path = os.path.join(DATASET_ROOT, folder)
        print(f"\nScanning folder: {folder}...")
        
        for sub in ["train", "test"]:
            sub_path = os.path.join(folder_path, sub)
            if not os.path.exists(sub_path):
                continue
                
            folders = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
            for f_dir in folders:
                f_dir_path = os.path.join(sub_path, f_dir)
                jpgs = [f for f in os.listdir(f_dir_path) if f.lower().endswith('.jpg')]
                
                # Apply step to subsample images
                jpgs_to_process = jpgs[::step] if step > 1 else jpgs
                
                for jpg in jpgs_to_process:
                    # Parse class index from filename (e.g. 'S1-A-1-27.jpg' -> parts[2] = '1')
                    parts = jpg.split('-')
                    if len(parts) < 3 or parts[2] not in CLASS_MAP:
                        continue
                        
                    letter_char = CLASS_MAP[parts[2]]
                    dest_dir = os.path.join(OUTPUT_ROOT, letter_char)
                    
                    img_path = os.path.join(f_dir_path, jpg)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Detect hands
                    results = detector.detect(img)
                    features = utils.extract_features(results)
                    
                    # Check if hands were detected (features not all zeros)
                    if not np.all(features == 0):
                        output_path = os.path.join(dest_dir, f"{sub}_{f_dir}_{jpg.replace('.jpg', '.npy')}")
                        np.save(output_path, features)
                        counts[letter_char] += 1
                        total_processed += 1
                    else:
                        skipped_count += 1
                        
                    if total_processed % 200 == 0 and total_processed > 0:
                        print(f"  Processed {total_processed} images so far...")
                        
    print(f"\n=== Keypoint Extraction Complete ===")
    print(f"Saved counts per letter: {counts}")
    print(f"Skipped {skipped_count} invalid frames (no hand detected).")
    
    # Train static letters model on new coordinates
    train_models.train_static_letters()

if __name__ == '__main__':
    main()
