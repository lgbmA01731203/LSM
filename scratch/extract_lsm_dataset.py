import os
import sys
import cv2
import numpy as np
import pandas as pd

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpu_hand_detector import GPUHandDetector
import utils

DATASET_DIR = r"C:\Users\capa_\OneDrive\Desktop\LSM\Mexican sign language dataset\MSLwords1"
OUTPUT_DIR = os.path.join("data", "words")

def sanitize_label(label):
    if not isinstance(label, str):
        return "unknown"
    label = label.strip().lower()
    
    # Specific cleanups for common character issues in classes sheet
    label = label.replace("caf", "cafe")
    label = label.replace("michoacn", "michoacan")
    label = label.replace("quertaro", "queretaro")
    label = label.replace("san luis potos", "san_luis_potosi")
    label = label.replace("nuevo len", "nuevo_leon")
    
    # Standard cleanups
    label = label.replace(" ", "_")
    label = label.replace("(", "_").replace(")", "")
    label = label.replace("/", "_")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace(",", "")
    return label

def resample_sequence(sequence, target_len=30):
    N = len(sequence)
    if N == 0:
        return np.zeros((target_len, 126), dtype=np.float32)
    
    sequence = np.array(sequence, dtype=np.float32)
    if N == target_len:
        return sequence
        
    indices = np.linspace(0, N - 1, target_len)
    resampled = np.zeros((target_len, 126), dtype=np.float32)
    for i, idx in enumerate(indices):
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        weight = idx - low
        if low == high:
            resampled[i] = sequence[low]
        else:
            resampled[i] = (1 - weight) * sequence[low] + weight * sequence[high]
    return resampled

def main():
    print("=== MSL Dataset Keypoint Extractor ===")
    
    xlsx_path = os.path.join(DATASET_DIR, "classes.xlsx")
    if not os.path.exists(xlsx_path):
        print(f"[ERROR] Excel file not found at: {xlsx_path}")
        return
        
    # Read classes excel
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns] # Strip spaces from column names
    
    # Map class index -> label
    class_map = {}
    for idx, row in df.iterrows():
        try:
            class_num = int(row['Class number'])
            word = row['Word']
            if pd.isna(word):
                word = f"class_{class_num}"
            class_map[class_num] = sanitize_label(word)
        except Exception as e:
            print(f"Error reading row {idx}: {e}")
            
    print(f"Mapped {len(class_map)} word classes from Excel.")
    
    # Initialize GPU Hand Detector
    detector = GPUHandDetector(palm_model_path="palm_detection.onnx", hand_model_path="handpose_estimation_mediapipe_2023feb.onnx")
    if not detector.load():
        print("[ERROR] Failed to load GPUHandDetector. Exiting.")
        return
        
    # Process each class
    extracted_total = 0
    skipped_total = 0
    
    # Sort classes to process in order
    class_nums = sorted(list(class_map.keys()))
    
    for i, class_num in enumerate(class_nums):
        class_folder_name = f"{class_num:03d}"
        class_path = os.path.join(DATASET_DIR, class_folder_name)
        word_label = class_map[class_num]
        
        if not os.path.exists(class_path):
            print(f"Directory {class_path} not found. Skipping.")
            continue
            
        dest_word_dir = os.path.join(OUTPUT_DIR, word_label)
        os.makedirs(dest_word_dir, exist_ok=True)
        
        # Find take directories inside class path
        takes = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        
        print(f"[{i+1}/{len(class_nums)}] Processing class '{word_label}' (folder: {class_folder_name}) with {len(takes)} takes...")
        
        for take in takes:
            take_path = os.path.join(class_path, take)
            # Find all .jpg files inside take path
            jpgs = sorted([f for f in os.listdir(take_path) if f.lower().endswith('.jpg')])
            
            if not jpgs:
                continue
                
            sequence_features = []
            for jpg in jpgs:
                img_path = os.path.join(take_path, jpg)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Detect hands
                results = detector.detect(img)
                features = utils.extract_features(results)
                sequence_features.append(features)
                
            if len(sequence_features) > 0:
                # Resample sequence to 30 frames
                resampled = resample_sequence(sequence_features, target_len=30)
                # Save numpy array
                output_filename = f"take_{take}.npy"
                np.save(os.path.join(dest_word_dir, output_filename), resampled)
                extracted_total += 1
            else:
                skipped_total += 1
                
        if (i + 1) % 10 == 0:
            print(f"--- Progress Check: Mapped {extracted_total} sequences so far ---")
            
    print(f"\n=== Extraction Complete ===")
    print(f"Successfully extracted {extracted_total} sequences.")
    print(f"Skipped/Empty: {skipped_total} sequences.")

if __name__ == '__main__':
    main()
