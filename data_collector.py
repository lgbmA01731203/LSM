import os
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
import utils
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_PATH = os.path.join('data')
STATIC_LETTERS_PATH = os.path.join(DATA_PATH, 'static_letters')
DYNAMIC_LETTERS_PATH = os.path.join(DATA_PATH, 'dynamic_letters')
WORDS_PATH = os.path.join(DATA_PATH, 'words')
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Create folders if they don't exist
for path in [STATIC_LETTERS_PATH, DYNAMIC_LETTERS_PATH, WORDS_PATH]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class LSMDataCollector:
    def __init__(self, mode='static_letter', label='unknown'):
        self.mode = mode  # 'static_letter', 'dynamic_letter', or 'word'
        self.label = label
        self.sequence_length = 30  # number of frames for dynamic sequences
        
    def get_class_dir(self):
        if self.mode == 'static_letter':
            root = STATIC_LETTERS_PATH
        elif self.mode == 'dynamic_letter':
            root = DYNAMIC_LETTERS_PATH
        else:
            root = WORDS_PATH
        class_dir = os.path.join(root, self.label)
        os.makedirs(class_dir, exist_ok=True)
        return class_dir
        
    def save_static_sample(self, features):
        """
        Saves a single frame's 126-dimensional features.
        """
        class_dir = self.get_class_dir()
        filename = f"{int(time.time() * 1000)}.npy"
        filepath = os.path.join(class_dir, filename)
        np.save(filepath, features)
        return filepath

    def save_dynamic_sample(self, sequence):
        """
        Saves a 30-frame sequence (shape 30x126).
        """
        class_dir = self.get_class_dir()
        filename = f"{int(time.time() * 1000)}.npy"
        filepath = os.path.join(class_dir, filename)
        np.save(filepath, np.array(sequence))
        return filepath

def run_cmdline_collector():
    """
    Optional command line interface for recording datasets easily.
    """
    print("=== LSM Data Collector ===")
    mode = input("Enter mode ('static_letter', 'dynamic_letter', or 'word'): ").strip().lower()
    if mode not in ['static_letter', 'dynamic_letter', 'word']:
        print("Invalid mode. Exiting.")
        return
        
    label = input("Enter gesture label (e.g. 'A', 'hola', 'gracias'): ").strip().lower()
    if not label:
        print("Invalid label. Exiting.")
        return
        
    collector = LSMDataCollector(mode=mode, label=label)
    
    # 1. Ensure model exists
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download finished!")
        
    # 2. Setup detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # 3. Setup OpenCV Video Capture
    cap = cv2.VideoCapture(0)
    
    print("\nPress 's' to capture a sample (or start sequence capture).")
    print("Press 'q' to quit.")
    
    sequence_length = 30
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with Tasks API HandLandmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        
        # Draw skeleton connections
        frame_drawn = utils.draw_landmarks(frame.copy(), results)
        
        cv2.putText(frame_drawn, f"Mode: {mode.upper()}  Label: {label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count existing samples
        num_samples = len(os.listdir(collector.get_class_dir()))
        cv2.putText(frame_drawn, f"Saved Samples: {num_samples}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Data Collector - Press 's' to save, 'q' to quit", frame_drawn)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            features = utils.extract_features(results)
            # Check if hands were detected (not all zeros)
            if np.all(features == 0):
                print("Warning: No hands detected! Gesture not saved.")
                continue
                
            if mode == 'static_letter':
                filepath = collector.save_static_sample(features)
                print(f"Captured static sample. Saved to {filepath}")
            else:
                # Capture a sequence of 30 frames
                print("Recording dynamic sequence...")
                sequence = []
                sequence.append(features)
                
                # Capture the remaining 29 frames
                frames_captured = 1
                while frames_captured < sequence_length:
                    ret_seq, frame_seq = cap.read()
                    if not ret_seq:
                        continue
                    frame_seq = cv2.flip(frame_seq, 1)
                    frame_seq_rgb = cv2.cvtColor(frame_seq, cv2.COLOR_BGR2RGB)
                    
                    mp_image_seq = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_seq_rgb)
                    results_seq = detector.detect(mp_image_seq)
                    
                    features_seq = utils.extract_features(results_seq)
                    sequence.append(features_seq)
                    frames_captured += 1
                    
                    # Draw and show progress
                    frame_seq_drawn = utils.draw_landmarks(frame_seq.copy(), results_seq)
                    cv2.putText(frame_seq_drawn, f"Recording: {frames_captured}/{sequence_length}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Data Collector - Press 's' to save, 'q' to quit", frame_seq_drawn)
                    cv2.waitKey(30)  # ~30 FPS capture rate
                    
                filepath = collector.save_dynamic_sample(sequence)
                print(f"Recorded dynamic sequence of {sequence_length} frames. Saved to {filepath}")
                
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == '__main__':
    run_cmdline_collector()
