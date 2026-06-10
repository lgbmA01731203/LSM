import os
import sys
import cv2
import numpy as np
import time

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpu_hand_detector import GPUHandDetector
import utils

def profile():
    print("=== Profiling Hand Detection Pipeline ===")
    
    # 1. Init detector
    detector = GPUHandDetector(
        palm_model_path="palm_detection.onnx",
        hand_model_path="handpose_estimation_mediapipe_2023feb.onnx"
    )
    if not detector.load():
        print("Failed to load detector.")
        return

    # Use a real camera or dummy image
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera, using dummy image with synthetic hand pattern.")
        # Create a dummy image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some white squares to simulate hands so palm detector might trigger
        cv2.rectangle(frame, (200, 200), (350, 350), (255, 255, 255), -1)
    else:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
    # Run 50 frames and profile
    times_palm = []
    times_hand = []
    times_preprocess = []
    times_postprocess = []
    times_draw = []
    times_total = []
    
    for i in range(50):
        t_start = time.time()
        
        # Preprocess
        t0 = time.time()
        h, w, _ = frame.shape
        image_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # simulating color conversion
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        input_blob, pad_bias = detector.palm_detector._preprocess(image_bgr)
        times_preprocess.append(time.time() - t0)
        
        # Palm detector forward
        t0 = time.time()
        output_blob = detector.palm_detector.sess.run(detector.palm_detector.output_names, {detector.palm_detector.input_name: input_blob})
        times_palm.append(time.time() - t0)
        
        # Palm postprocess
        t0 = time.time()
        palms = detector.palm_detector._postprocess(output_blob, np.array([w, h]), pad_bias)
        times_postprocess.append(time.time() - t0)
        
        # Force a mock palm to profile landmark model execution
        mock_palm = np.zeros(19)
        mock_palm[0:4] = [100, 100, 300, 300]
        mock_palm[4:18] = [150, 150, 160, 170, 170, 180, 180, 190, 190, 200, 200, 210, 210, 220]
        palms = [mock_palm]
        
        # Handpose
        t_hand_total = 0
        for palm in palms:
            t0 = time.time()
            handpose = detector.hand_detector.infer(image_bgr, palm)
            t_hand_total += (time.time() - t0)
        times_hand.append(t_hand_total)
        
        # Mock draw landmarks
        t0 = time.time()
        # Mock results object
        class MockResult:
            def __init__(self):
                self.hand_landmarks = []
                self.handedness = []
        res = MockResult()
        for palm in palms:
            res.hand_landmarks.append([None]*21)
        # draw_landmarks (mostly dummy)
        # utils.draw_landmarks(frame.copy(), res)
        times_draw.append(time.time() - t0)
        
        times_total.append(time.time() - t_start)
        
    if cap.isOpened():
        cap.release()
        
    print(f"Average times over 50 frames:")
    print(f"  Palm Preprocess  : {np.mean(times_preprocess)*1000:.2f} ms")
    print(f"  Palm Network Run : {np.mean(times_palm)*1000:.2f} ms")
    print(f"  Palm Postprocess : {np.mean(times_postprocess)*1000:.2f} ms")
    print(f"  Hand Pose Run    : {np.mean(times_hand)*1000:.2f} ms")
    print(f"  Draw landmarks   : {np.mean(times_draw)*1000:.2f} ms")
    print(f"  Total Pipeline   : {np.mean(times_total)*1000:.2f} ms ({1.0/np.mean(times_total):.2f} FPS)")

if __name__ == "__main__":
    profile()
