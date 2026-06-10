"""
gpu_hand_detector.py
====================
GPU-accelerated hand landmark detector using ONNX Runtime + CUDA.
Replaces MediaPipe HandLandmarker completely.
Wraps the MPPalmDet and MPHandPose classes from the OpenCV model zoo.
"""

import os
import sys

# Pre-load packaged NVIDIA CUDA/cuDNN DLLs into Windows DLL search path
try:
    import site
    site_dirs = site.getsitepackages()
    for sd in site_dirs:
        nvidia_root = os.path.join(sd, "nvidia")
        if os.path.exists(nvidia_root):
            for root, dirs, files in os.walk(nvidia_root):
                if any(f.endswith(".dll") for f in files):
                    os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]
                    try:
                        os.add_dll_directory(root)
                    except AttributeError:
                        pass
                    except Exception:
                        pass
except Exception:
    pass

import cv2
import numpy as np
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

class Category:
    """Mock MediaPipe Category class."""
    def __init__(self, category_name: str, score: float = 1.0):
        self.category_name = category_name
        self.display_name = category_name
        self.score = score

class Landmark:
    """Mock MediaPipe NormalizedLandmark class."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class GPUHandDetectorResult:
    """Mock MediaPipe HandLandmarkerResult class."""
    def __init__(self, hand_landmarks, handedness):
        self.hand_landmarks = hand_landmarks  # List of List of Landmark
        self.handedness = handedness  # List of List of Category

class GPUHandDetector:
    """
    Drop-in replacement for MediaPipe HandLandmarker.
    Uses ONNX Runtime with CUDA for fast GPU hand pose estimation.
    """
    def __init__(self, palm_model_path="palm_detection.onnx", hand_model_path="handpose_estimation_mediapipe_2023feb.onnx"):
        self.palm_model_path = palm_model_path
        self.hand_model_path = hand_model_path
        
        self.palm_detector = None
        self.hand_detector = None
        self._available = False
        
    def load(self):
        """Initializes ONNX models."""
        try:
            if not os.path.exists(self.palm_model_path) or not os.path.exists(self.hand_model_path):
                print(f"[GPUHandDetector] Model files missing. Ensure {self.palm_model_path} and {self.hand_model_path} are in workspace.")
                return False
                
            self.palm_detector = MPPalmDet(self.palm_model_path, scoreThreshold=0.4)
            self.hand_detector = MPHandPose(self.hand_model_path)
            self._available = True
            print("[GPUHandDetector] Initialized successfully on GPU.")
            return True
        except Exception as e:
            print(f"[GPUHandDetector] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    @property
    def available(self):
        return self._available

    def detect(self, image_bgr: np.ndarray):
        """
        Run palm detection + landmark estimation on a BGR image.
        
        Returns:
            GPUHandDetectorResult object containing lists of normalized hand_landmarks and handedness.
        """
        if not self._available:
            return GPUHandDetectorResult([], [])

        h, w, _ = image_bgr.shape

        # 1. Palm detection -> bounding boxes
        palms = self.palm_detector.infer(image_bgr)
        
        hand_landmarks_list = []
        handedness_list = []

        # Only process up to 2 palms to limit computation and prevent false positive latency
        for palm in palms[:2]:
            # 2. Landmark extraction
            handpose = self.hand_detector.infer(image_bgr, palm)
            if handpose is not None:
                # Extract screen landmarks [x1, y1, z1, x2, y2, z2, ...]
                # Screen landmarks are at handpose[4:67]
                landmarks_screen = handpose[4:67].reshape(21, 3)
                
                # Convert landmarks to normalized coordinates (0.0 to 1.0)
                landmarks = []
                for lm in landmarks_screen:
                    # x and y normalized by image width and height, z normalized by width
                    landmarks.append(Landmark(
                        x=lm[0] / w,
                        y=lm[1] / h,
                        z=lm[2] / w
                    ))
                hand_landmarks_list.append(landmarks)

                # Extract handedness score (handpose[-2])
                raw_handedness = handpose[-2]
                handedness_text = "Left" if raw_handedness <= 0.5 else "Right"
                handedness_list.append([Category(handedness_text, score=float(raw_handedness))])

        return GPUHandDetectorResult(hand_landmarks_list, handedness_list)

    def close(self):
        pass
