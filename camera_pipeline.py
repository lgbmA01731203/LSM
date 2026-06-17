import os
import cv2
import threading
import queue
import time
import urllib.request
import traceback
import numpy as np
import utils
from gpu_hand_detector import GPUHandDetector

PALM_MODEL_URL = "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx"
PALM_MODEL_PATH = "palm_detection.onnx"
POSE_MODEL_URL = "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
POSE_MODEL_PATH = "handpose_estimation_mediapipe_2023feb.onnx"

# Resolution fed to MediaPipe for detection
DETECT_W, DETECT_H = 256, 192

# Fixed display size sent to the GUI (avoids expensive per-frame rescaling in tkinter)
DISPLAY_W, DISPLAY_H = 640, 480


class LSMCameraPipeline:
    """
    3-thread pipeline:
      Thread 1 (camera)   — reads raw frames from webcam as fast as possible
      Thread 2 (detector) — pulls latest raw frame, runs MediaPipe, draws skeleton
      GUI thread          — reads the latest finished frame + features
    Camera and MediaPipe never block each other.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index

        self.running = False
        self._cam_thread = None
        self._det_thread = None

        # Single-slot "mailbox" between camera → detector  (always latest frame)
        self._raw_frame_slot = None
        self._raw_lock = threading.Lock()

        # Single-slot mailbox between detector → GUI
        self._out_frame = None      # RGB numpy array, resized for display
        self._out_features = np.zeros(126)
        self._out_lock = threading.Lock()

        self.max_num_hands = 2
        self.min_detection_confidence = 0.65
        self.min_tracking_confidence = 0.5
        
        self.stabilizer = utils.HandLandmarkStabilizer(alpha=0.3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_model_exists(self):
        # Check palm detection model
        if not os.path.exists(PALM_MODEL_PATH):
            print("Descargando modelo de Detección de Palma (ONNX)...")
            try:
                urllib.request.urlretrieve(PALM_MODEL_URL, PALM_MODEL_PATH)
                print("Modelo de Palma descargado.")
            except Exception as e:
                print(f"Error descargando modelo de Palma: {e}")
                
        # Check hand pose landmark model
        if not os.path.exists(POSE_MODEL_PATH):
            print("Descargando modelo de Landmark de Mano (ONNX)...")
            try:
                urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
                print("Modelo de Landmark descargado.")
            except Exception as e:
                print(f"Error descargando modelo de Landmark: {e}")

    def start(self):
        if not self.running:
            self.check_model_exists()
            self.running = True

            self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True, name="CameraReader")
            self._det_thread = threading.Thread(target=self._detector_loop, daemon=True, name="MPDetector")

            self._cam_thread.start()
            self._det_thread.start()

    def stop(self):
        self.running = False
        if self._cam_thread:
            self._cam_thread.join(timeout=2.0)
        if self._det_thread:
            self._det_thread.join(timeout=2.0)

    def get_latest(self):
        """Returns (frame_rgb_numpy | None, features_numpy)."""
        with self._out_lock:
            return self._out_frame, self._out_features.copy()

    # ------------------------------------------------------------------
    # Thread 1 — camera reader (no MediaPipe here at all)
    # ------------------------------------------------------------------

    def _camera_loop(self):
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always freshest frame

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(frame, 1)  # selfie mirror

                # Store latest frame — detector picks it up whenever it's free
                with self._raw_lock:
                    self._raw_frame_slot = frame

            cap.release()
        except Exception as e:
            _log_error("camera_error.log", e)

    # ------------------------------------------------------------------
    # Thread 2 — MediaPipe detector (runs as fast as it can, independently)
    # ------------------------------------------------------------------

    def _detector_loop(self):
        try:
            detector = GPUHandDetector(palm_model_path=PALM_MODEL_PATH, hand_model_path=POSE_MODEL_PATH)
            if not detector.load():
                print("[CameraPipeline] Failed to load GPUHandDetector. Exiting thread.")
                return

            last_frame_id = None  # avoid reprocessing the same frame
            last_results  = None
            last_features = np.zeros(126)

            while self.running:
                # Grab latest raw frame
                with self._raw_lock:
                    frame = self._raw_frame_slot

                if frame is None or id(frame) == last_frame_id:
                    time.sleep(0.005)   # wait for a new frame
                    continue

                last_frame_id = id(frame)

                # ---- Run GPU Detector directly on the BGR frame ----
                last_results  = detector.detect(frame)
                
                # Smooth the landmarks in-place
                last_results  = self.stabilizer.smooth(last_results)
                
                last_features = utils.extract_features(last_results)

                # ---- Draw skeleton on the FULL-RES display frame ----
                drawn = utils.draw_landmarks(frame, last_results)

                # ---- Resize to fixed display size & convert to RGB ----
                display = cv2.resize(drawn, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                # Publish results to GUI
                with self._out_lock:
                    self._out_frame    = rgb
                    self._out_features = last_features

            detector.close()

        except Exception as e:
            _log_error("camera_error.log", e)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _log_error(path, exc):
    with open(path, "w") as f:
        f.write(f"Exception in pipeline thread:\n")
        import traceback as tb
        tb.print_exc(file=f)
    print(f"Pipeline error: {exc}")
