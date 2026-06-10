"""
extract_dynamic_videos.py
=========================
Extracts 126-dimensional sequences from LSM dynamic gesture videos.
Resamples each video to exactly 30 frames for LSTM training.
"""

import os
import glob
import cv2
import numpy as np
import uuid
import sys
import utils
from gpu_hand_detector import GPUHandDetector

DATA_OUTPUT_DIR = os.path.join('data', 'dynamic_letters')
TARGET_FRAMES = 30

def extract_features_from_video(detector, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    sequence = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result = detector.detect(frame)
        if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
            features = utils.extract_features(result)
            sequence.append(features)
        else:
            # If no hand detected, use a zero vector or copy the previous valid frame
            if len(sequence) > 0:
                sequence.append(sequence[-1])
            else:
                sequence.append(np.zeros(126, dtype=np.float32))
                
    cap.release()
    
    if len(sequence) == 0:
        return None
        
    sequence = np.array(sequence)
    
    # Resample to exactly TARGET_FRAMES
    num_frames = len(sequence)
    if num_frames == 0:
        return None
        
    # Create an array of indices to sample
    indices = np.linspace(0, num_frames - 1, TARGET_FRAMES).astype(int)
    resampled_sequence = sequence[indices]
    
    return resampled_sequence

def process_directory(detector, directory, is_frontal=True):
    total = 0
    failed = 0
    
    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return
        
    print(f"\nProcessing {directory}...")
    
    if is_frontal:
        # Format: S1-J-frontal-1.mp4
        mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
        for f in mp4_files:
            basename = os.path.basename(f)
            parts = basename.split('-')
            if len(parts) >= 2:
                cls_raw = parts[1].lower()
                # Handle special characters for ñ
                if cls_raw not in ['j', 'k', 'q', 'x', 'z']:
                    cls = 'ñ'
                else:
                    cls = cls_raw
                    
                seq = extract_features_from_video(detector, f)
                if seq is not None:
                    out_dir = os.path.join(DATA_OUTPUT_DIR, cls)
                    os.makedirs(out_dir, exist_ok=True)
                    np.save(os.path.join(out_dir, f"video_{uuid.uuid4().hex[:8]}.npy"), seq)
                    total += 1
                else:
                    failed += 1
    else:
        # Profile format: Subfolders J, K, etc. Inside: S1-J-perfil-1.mp4
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        for sub in subdirs:
            cls_raw = sub.lower()
            if cls_raw not in ['j', 'k', 'q', 'x', 'z']:
                cls = 'ñ'
            else:
                cls = cls_raw
                
            mp4_files = glob.glob(os.path.join(directory, sub, '*.mp4'))
            for f in mp4_files:
                seq = extract_features_from_video(detector, f)
                if seq is not None:
                    out_dir = os.path.join(DATA_OUTPUT_DIR, cls)
                    os.makedirs(out_dir, exist_ok=True)
                    np.save(os.path.join(out_dir, f"video_{uuid.uuid4().hex[:8]}.npy"), seq)
                    total += 1
                else:
                    failed += 1
                    
    print(f"Extracted {total} videos. Failed {failed} videos.")

def main():
    print("Initializing GPU Hand Detector...")
    detector = GPUHandDetector()
    if not detector.load():
        print("[ERROR] Could not load GPU hand detector.")
        sys.exit(1)
        
    dir_frontal = os.path.join('MSL-dynamic-signs-frontal-view', 'MSL-dynamic-signs', 'train')
    dir_profile = os.path.join('MSL-dynamic-signs-profile', 'MSL dynamic-profile-signs')
    
    process_directory(detector, dir_frontal, is_frontal=True)
    process_directory(detector, dir_profile, is_frontal=False)
    
    print("\nExtraction Complete.")
    
if __name__ == '__main__':
    main()
