import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import utils

from train_models import LSMLSTMModel

class LSMTranslator:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Static Letters Classifier
        self.static_classifier = None
        self.static_classes = []
        
        # 2. Dynamic Letters Model
        self.dynamic_letters_model = None
        self.dynamic_letters_classes = []
        
        # 3. Words Model
        self.words_model = None
        self.words_classes = []
        
        # Sequencer for dynamic coordinates (holds last 30 frames of 126-dim features)
        self.sequence_len = 30
        self.sequence = deque(maxlen=self.sequence_len)
        
        # Debouncing/smoothing states
        self.letter_history = deque(maxlen=10) # last 10 letter predictions
        self.last_committed_letter = None
        self.no_hand_counter = 0
        
        # Per-hand carry-over for occlusion handling
        self.last_valid_left = None
        self.last_valid_right = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0
        
        # Lag elimination (LSTM decimation)
        self.frame_counter = 0
        self.last_lstm_word_pred = (None, None, 0.0)
        self.last_lstm_dyn_pred = (None, None, 0.0)
        
        self.dynamic_cooldown = 0 # frames to wait before predicting another word
        
        self.load_models()
        
    def load_models(self):
        """
        Loads the three classifiers if they exist.
        """
        # A. Load Static Letters Model
        static_model_path = os.path.join(self.models_dir, 'static_letters_classifier.joblib')
        static_classes_path = os.path.join(self.models_dir, 'static_letters_classes.json')
        if os.path.exists(static_model_path) and os.path.exists(static_classes_path):
            try:
                self.static_classifier = joblib.load(static_model_path)
                # Force n_jobs=1 for real-time inference to avoid thread thrashing lag
                if hasattr(self.static_classifier, 'calibrated_classifiers_'):
                    for clf in self.static_classifier.calibrated_classifiers_:
                        if hasattr(clf.estimator, 'n_jobs'):
                            clf.estimator.n_jobs = 1
                elif hasattr(self.static_classifier, 'n_jobs'):
                    self.static_classifier.n_jobs = 1
                
                with open(static_classes_path, 'r') as f:
                    self.static_classes = json.load(f)
                print("Loaded static letters classifier.")
            except Exception as e:
                print(f"Error loading static letters classifier: {e}")
                
        # B. Load Dynamic Letters Model
        dyn_let_model_path = os.path.join(self.models_dir, 'dynamic_letters_model.pth')
        dyn_let_classes_path = os.path.join(self.models_dir, 'dynamic_letters_classes.json')
        if os.path.exists(dyn_let_model_path) and os.path.exists(dyn_let_classes_path):
            try:
                with open(dyn_let_classes_path, 'r') as f:
                    self.dynamic_letters_classes = json.load(f)
                
                self.dynamic_letters_model = LSMLSTMModel(
                    input_dim=126, 
                    hidden_dim=64, 
                    num_layers=2, 
                    num_classes=len(self.dynamic_letters_classes)
                )
                self.dynamic_letters_model.load_state_dict(torch.load(dyn_let_model_path, map_location=self.device))
                self.dynamic_letters_model.to(self.device)
                self.dynamic_letters_model.eval()
                print("Loaded dynamic letters model (LSTM).")
            except Exception as e:
                print(f"Error loading dynamic letters model: {e}")
                
        # C. Load Words Model
        words_model_path = os.path.join(self.models_dir, 'words_model.pth')
        words_classes_path = os.path.join(self.models_dir, 'words_classes.json')
        if os.path.exists(words_model_path) and os.path.exists(words_classes_path):
            try:
                with open(words_classes_path, 'r') as f:
                    self.words_classes = json.load(f)
                
                self.words_model = LSMLSTMModel(
                    input_dim=126, 
                    hidden_dim=64, 
                    num_layers=2, 
                    num_classes=len(self.words_classes)
                )
                self.words_model.load_state_dict(torch.load(words_model_path, map_location=self.device))
                self.words_model.to(self.device)
                self.words_model.eval()
                print("Loaded words model (LSTM).")
            except Exception as e:
                print(f"Error loading words model: {e}")

    # ------------------------------------------------------------------ #
    #                       GEOMETRIC HELPER METHODS                      #
    # ------------------------------------------------------------------ #

    def _get_finger_ratios(self, features):
        left_hand = features[0:63]
        right_hand = features[63:126]
        hand = left_hand if not np.all(left_hand == 0) else right_hand
        if np.all(hand == 0):
            return None, None, None
            
        coords = hand.reshape(21, 3)
        fingers = [
            (8, 7, 6, 5),     # Index
            (12, 11, 10, 9),  # Middle
            (16, 15, 14, 13), # Ring
            (20, 19, 18, 17)  # Pinky
        ]
        ratios = []
        for tip, dip, pip, mcp in fingers:
            d_mcp_tip = np.linalg.norm(coords[tip] - coords[mcp])
            sum_segments = (np.linalg.norm(coords[pip] - coords[mcp]) + 
                            np.linalg.norm(coords[dip] - coords[pip]) + 
                            np.linalg.norm(coords[tip] - coords[dip]))
            ratio = d_mcp_tip / sum_segments if sum_segments > 0 else 0.0
            ratios.append(ratio)
            
        # Thumb straightness
        d_thumb_tip_mcp = np.linalg.norm(coords[4] - coords[1])
        sum_thumb_segments = (np.linalg.norm(coords[2] - coords[1]) + 
                              np.linalg.norm(coords[3] - coords[2]) + 
                              np.linalg.norm(coords[4] - coords[3]))
        thumb_ratio = d_thumb_tip_mcp / sum_thumb_segments if sum_thumb_segments > 0 else 0.0
        
        return ratios, thumb_ratio, coords

    def _palm_normal(self, coords):
        """
        Returns the palm normal vector (unit) from cross product
        of (wrist→index_mcp) × (wrist→pinky_mcp).
        Also returns the raw normal_z component (signed).
        """
        v1 = coords[5] - coords[0]   # wrist → index MCP
        v2 = coords[17] - coords[0]  # wrist → pinky MCP
        normal_vec = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal_vec)
        if norm_val > 0:
            normal_unit = normal_vec / norm_val
        else:
            normal_unit = np.array([0.0, 0.0, 0.0])
        return normal_unit, normal_unit[2]

    def _palm_normal_z(self, coords):
        """Returns abs(normal_z) — how much the palm faces the camera."""
        _, nz = self._palm_normal(coords)
        return abs(nz)

    def _hand_rotation_degrees(self, coords):
        """
        0° = palm fully facing camera, 90° = palm fully sideways.
        Uses arccos of |normal_z|.
        """
        abs_nz = self._palm_normal_z(coords)
        return np.degrees(np.arccos(np.clip(abs_nz, 0.0, 1.0)))

    def _thumb_to_fingertip_dists(self, coords):
        """
        Distances from thumb tip (4) to each fingertip (8, 12, 16, 20).
        Returns [d_thumb_index, d_thumb_middle, d_thumb_ring, d_thumb_pinky].
        """
        thumb_tip = coords[4]
        return [
            np.linalg.norm(thumb_tip - coords[8]),
            np.linalg.norm(thumb_tip - coords[12]),
            np.linalg.norm(thumb_tip - coords[16]),
            np.linalg.norm(thumb_tip - coords[20]),
        ]

    def _inter_finger_angles(self, coords):
        """
        Angles (radians) between adjacent fingertip vectors from the wrist.
        Returns [thumb-index, index-middle, middle-ring, ring-pinky].
        """
        tips = [4, 8, 12, 16, 20]
        angles = []
        for i in range(len(tips) - 1):
            va = coords[tips[i]] - coords[0]
            vb = coords[tips[i + 1]] - coords[0]
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            if na > 0 and nb > 0:
                cos_a = np.dot(va, vb) / (na * nb)
                angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            else:
                angles.append(0.0)
        return angles

    def _palm_finger_alignment(self, coords):
        """
        Alignment (cosine similarity) of palm direction (wrist→middle_MCP)
        with middle finger direction (middle_MCP→middle_tip).
        """
        v_palm = coords[9] - coords[0]
        v_finger = coords[12] - coords[9]
        n_palm = np.linalg.norm(v_palm)
        n_finger = np.linalg.norm(v_finger)
        if n_palm > 0 and n_finger > 0:
            return np.dot(v_palm / n_palm, v_finger / n_finger)
        return 0.0

    def _finger_tip_distance(self, coords, tip_a, tip_b):
        """Euclidean distance between two landmark indices."""
        return np.linalg.norm(coords[tip_a] - coords[tip_b])

    def _thumb_position_relative(self, coords):
        """
        Determines where the thumb tip is relative to the finger plane.
        Returns 'side', 'front', 'under', or 'between' based on geometry.
        
        - 'side': thumb tip is beside the index MCP (A-like)
        - 'front': thumb tip is in front of the curled fingers (E/S-like)
        - 'under': thumb tip is below the MCP plane (M/N/T-like)
        - 'between': thumb tip is between index and middle (T-like)
        """
        thumb_tip = coords[4]
        index_mcp = coords[5]
        middle_mcp = coords[9]
        index_pip = coords[6]
        middle_pip = coords[10]
        
        # Check if thumb is between index and middle
        mid_point_idx_mid = (index_mcp + middle_mcp) / 2.0
        d_thumb_between = np.linalg.norm(thumb_tip - mid_point_idx_mid)
        d_idx_mid_mcps = np.linalg.norm(index_mcp - middle_mcp)
        
        if d_thumb_between < d_idx_mid_mcps * 0.8:
            # Thumb is close to the gap between index and middle
            # Check if it's tucked (below PIP line)
            pip_midpoint = (index_pip + middle_pip) / 2.0
            if thumb_tip[1] > pip_midpoint[1]:  # y increases downward in normalized coords
                return 'between'
        
        # Check if thumb is below MCP plane (under fingers for M/N)
        mcp_plane_y = np.mean([coords[5][1], coords[9][1], coords[13][1], coords[17][1]])
        pip_plane_y = np.mean([coords[6][1], coords[10][1], coords[14][1], coords[18][1]])
        if thumb_tip[1] > pip_plane_y:
            return 'under'
        
        # Check side vs front using palm normal
        _, normal_z = self._palm_normal(coords)
        v_wrist_thumb = coords[4] - coords[0]
        v_wrist_index_mcp = coords[5] - coords[0]
        
        # Project thumb vector onto palm normal direction
        normal_unit, _ = self._palm_normal(coords)
        if np.linalg.norm(normal_unit) > 0:
            proj = np.dot(v_wrist_thumb, normal_unit)
            if abs(proj) > 0.15:
                return 'front'
        
        return 'side'

    def _finger_orientation_y(self, coords):
        """
        Returns the average y-component of extended finger directions.
        Positive = fingers point downward, negative = upward.
        Used to detect P (hand pointing down).
        """
        # Use middle finger as representative
        finger_dir = coords[12] - coords[9]
        n = np.linalg.norm(finger_dir)
        if n > 0:
            return finger_dir[1] / n
        return 0.0

    def _hand_pointing_angle(self, coords):
        """
        Returns the 2D angle (in degrees) of the hand pointing direction (wrist to middle MCP)
        relative to the UP direction.
        0° = UP, 90° = SIDEWAYS (left or right), 180° = DOWN.
        """
        v = coords[9] - coords[0] # wrist to middle MCP
        v_2d = v[:2]
        n = np.linalg.norm(v_2d)
        if n == 0: return 0.0
        v_2d = v_2d / n
        cos_theta = -v_2d[1]
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # ------------------------------------------------------------------ #
    #                     RULE-BASED CLASSIFIER (22 letters)              #
    # ------------------------------------------------------------------ #

    def classify_by_rules(self, features):
        """
        Classifies the hand shape into one of 22 static LSM letters using
        strict geometric rules derived from LSM research.
        
        Returns (predicted_class, confidence) or (None, 0.0).
        
        Letters covered:
          Palm facing camera (rotation < 50°):
            A, B, D, E, F, I, K, L, M, N, O, R, S, T, U, V, W, X, Y
          Palm sideways (rotation >= 50°):
            C, G, H, P
        """
        ratios, thumb_ratio, coords = self._get_finger_ratios(features)
        if ratios is None:
            return None, 0.0

        # Unpack individual finger ratios
        idx_r, mid_r, ring_r, pinky_r = ratios

        # Palm orientation
        rotation_deg = self._hand_rotation_degrees(coords)
        alignment = self._palm_finger_alignment(coords)

        # Thumb-to-fingertip distances
        thumb_dists = self._thumb_to_fingertip_dists(coords)
        d_thumb_idx, d_thumb_mid, d_thumb_ring, d_thumb_pinky = thumb_dists

        # Inter-finger angles [thumb-index, index-middle, middle-ring, ring-pinky]
        angles = self._inter_finger_angles(coords)
        a_thumb_idx, a_idx_mid, a_mid_ring, a_ring_pinky = angles

        # Hand pointing angle (0=UP, 90=SIDEWAYS)
        pointing_up_angle = self._hand_pointing_angle(coords)

        # Aggregate stats
        num_straight = sum(1 for r in ratios if r > 0.85)
        num_curled = sum(1 for r in ratios if r < 0.78)
        num_tight_curled = sum(1 for r in ratios if r < 0.65)
        mean_ratio = np.mean(ratios)

        # Index-middle tip distance (for R vs U)
        d_idx_mid_tips = self._finger_tip_distance(coords, 8, 12)

        best_letter = None
        best_conf = 0.0

        # ============================================================== #
        #  PALM FACING CAMERA  (rotation < 50°)                           #
        # ============================================================== #
        if rotation_deg < 50.0:

            # --- A: All 4 curled, thumb extended to SIDE ---
            if all(r < 0.78 for r in ratios) and thumb_ratio >= 0.68:
                # Thumb should be to the side, not in front
                thumb_pos = self._thumb_position_relative(coords)
                if thumb_pos == 'side':
                    avg_curl = np.mean(ratios)
                    conf = 0.70 + 0.25 * (1.0 - avg_curl / 0.78)
                    conf += 0.05 * min(1.0, (thumb_ratio - 0.68) / 0.3)
                    if best_conf < conf:
                        best_letter, best_conf = 'a', min(0.95, conf)

            # --- E: All 4 curled, thumb curled IN FRONT ---
            if all(r < 0.78 for r in ratios) and thumb_ratio < 0.68:
                avg_curl = np.mean(ratios)
                conf = 0.70 + 0.25 * (1.0 - avg_curl / 0.78)
                conf += 0.05 * (1.0 - thumb_ratio / 0.68)
                if best_conf < conf:
                    best_letter, best_conf = 'e', min(0.95, conf)

            # --- S: Tight fist, thumb wraps OVER front of fingers ---
            if all(r < 0.65 for r in ratios) and thumb_ratio >= 0.55:
                thumb_pos = self._thumb_position_relative(coords)
                # S: thumb is in front, wrapping over the fingers
                # Check thumb tip is near middle finger PIP area
                d_thumb_mid_pip = np.linalg.norm(coords[4] - coords[10])
                if thumb_pos == 'front' or d_thumb_mid_pip < 0.25:
                    avg_curl = np.mean(ratios)
                    conf = 0.68 + 0.25 * (1.0 - avg_curl / 0.65)
                    if best_conf < conf:
                        best_letter, best_conf = 's', min(0.95, conf)

            # --- B: All 4 straight, aligned, thumb folded ---
            if all(r > 0.82 for r in ratios) and alignment > 0.78 and thumb_ratio < 0.88:
                avg_str = np.mean(ratios)
                conf = 0.70 + 0.20 * (avg_str - 0.82) / 0.18
                conf += 0.10 * (alignment - 0.78) / 0.22
                if best_conf < conf:
                    best_letter, best_conf = 'b', min(0.95, conf)

            # --- D: Index straight, others curled, thumb touches middle ---
            if idx_r > 0.85 and all(r < 0.70 for r in [mid_r, ring_r, pinky_r]):
                if d_thumb_mid < 0.25:
                    conf = 0.70
                    conf += 0.10 * min(1.0, (idx_r - 0.85) / 0.15)
                    conf += 0.10 * max(0.0, 1.0 - d_thumb_mid / 0.25)
                    conf += 0.05 * (1.0 - np.mean([mid_r, ring_r, pinky_r]) / 0.70)
                    if best_conf < conf:
                        best_letter, best_conf = 'd', min(0.95, conf)

            # --- F: Middle+Ring+Pinky straight, Index curled, thumb touches index tip ---
            if all(r > 0.85 for r in [mid_r, ring_r, pinky_r]) and idx_r < 0.78:
                if d_thumb_idx < 0.18:
                    conf = 0.72
                    conf += 0.10 * max(0.0, 1.0 - d_thumb_idx / 0.18)
                    conf += 0.08 * min(1.0, (np.mean([mid_r, ring_r, pinky_r]) - 0.85) / 0.15)
                    if best_conf < conf:
                        best_letter, best_conf = 'f', min(0.95, conf)

            # --- I: Only pinky straight, others curled ---
            if pinky_r > 0.85 and all(r < 0.70 for r in [idx_r, mid_r, ring_r]):
                # Make sure thumb is not significantly extended (otherwise could be Y)
                if a_thumb_idx < 1.0 or thumb_ratio < 0.75:
                    conf = 0.72
                    conf += 0.10 * min(1.0, (pinky_r - 0.85) / 0.15)
                    conf += 0.08 * (1.0 - np.mean([idx_r, mid_r, ring_r]) / 0.70)
                    if best_conf < conf:
                        best_letter, best_conf = 'i', min(0.95, conf)

            # --- Y: Thumb + Pinky extended, index+middle+ring curled ---
            if pinky_r > 0.80 and thumb_ratio > 0.65 and a_thumb_idx >= 0.50:
                if all(r < 0.70 for r in [idx_r, mid_r, ring_r]):
                    conf = 0.72
                    conf += 0.08 * min(1.0, (pinky_r - 0.80) / 0.20)
                    conf += 0.08 * min(1.0, (thumb_ratio - 0.65) / 0.35)
                    conf += 0.07 * (1.0 - np.mean([idx_r, mid_r, ring_r]) / 0.70)
                    if best_conf < conf:
                        best_letter, best_conf = 'y', min(0.95, conf)

            # --- L: Index straight, thumb extended perpendicular, clear L-shape ---
            if idx_r > 0.85 and thumb_ratio > 0.80 and pointing_up_angle < 60.0:
                if all(r < 0.70 for r in [mid_r, ring_r, pinky_r]):
                    # L-shape: angle between thumb and index should be roughly 60-120°
                    if 0.35 < a_thumb_idx < 2.20:
                        conf = 0.72
                        conf += 0.08 * min(1.0, (idx_r - 0.85) / 0.15)
                        conf += 0.08 * min(1.0, (thumb_ratio - 0.80) / 0.20)
                        # Penalize if angle is too narrow or too wide
                        angle_quality = 1.0 - abs(a_thumb_idx - 1.20) / 1.20
                        conf += 0.07 * max(0.0, angle_quality)
                        if best_conf < conf:
                            best_letter, best_conf = 'l', min(0.95, conf)

            # --- K: Index straight, middle angled, thumb touching middle ---
            if idx_r > 0.85 and 0.55 < mid_r < 0.85:
                if all(r < 0.70 for r in [ring_r, pinky_r]):
                    if d_thumb_mid < 0.45 and thumb_ratio > 0.45:
                        conf = 0.68
                        conf += 0.08 * min(1.0, (idx_r - 0.85) / 0.15)
                        conf += 0.08 * (1.0 - abs(mid_r - 0.70) / 0.30)
                        conf += 0.06 * max(0.0, 1.0 - d_thumb_mid / 0.45)
                        if best_conf < conf:
                            best_letter, best_conf = 'k', min(0.95, conf)

            # --- U: Index + Middle straight, TOGETHER, others curled ---
            if all(r > 0.85 for r in [idx_r, mid_r]) and pointing_up_angle < 60.0:
                if all(r < 0.70 for r in [ring_r, pinky_r]):
                    if a_idx_mid < 0.12:
                        conf = 0.72
                        conf += 0.08 * min(1.0, (np.mean([idx_r, mid_r]) - 0.85) / 0.15)
                        conf += 0.10 * max(0.0, (0.12 - a_idx_mid) / 0.12)
                        if best_conf < conf:
                            best_letter, best_conf = 'u', min(0.95, conf)

            # --- V: Index + Middle straight, SPREAD, others curled ---
            if all(r > 0.85 for r in [idx_r, mid_r]) and pointing_up_angle < 60.0:
                if all(r < 0.70 for r in [ring_r, pinky_r]):
                    if a_idx_mid > 0.20:
                        conf = 0.72
                        conf += 0.08 * min(1.0, (np.mean([idx_r, mid_r]) - 0.85) / 0.15)
                        conf += 0.10 * min(1.0, (a_idx_mid - 0.20) / 0.40)
                        if best_conf < conf:
                            best_letter, best_conf = 'v', min(0.95, conf)

            # --- R: Index + Middle extended, CROSSED (very close tips) ---
            if all(r > 0.80 for r in [idx_r, mid_r]):
                if all(r < 0.70 for r in [ring_r, pinky_r]):
                    if a_idx_mid < 0.18 and d_idx_mid_tips < 0.20:
                        conf = 0.70
                        conf += 0.10 * max(0.0, (0.20 - d_idx_mid_tips) / 0.20)
                        conf += 0.10 * max(0.0, (0.18 - a_idx_mid) / 0.18)
                        if best_conf < conf:
                            best_letter, best_conf = 'r', min(0.95, conf)

            # --- W: Index + Middle + Ring straight, spread, pinky curled ---
            if all(r > 0.85 for r in [idx_r, mid_r, ring_r]) and pinky_r < 0.70:
                conf = 0.70
                conf += 0.08 * min(1.0, (np.mean([idx_r, mid_r, ring_r]) - 0.85) / 0.15)
                conf += 0.07 * (1.0 - pinky_r / 0.70)
                # Check some spread between fingers
                if a_idx_mid > 0.08 or a_mid_ring > 0.08:
                    conf += 0.05
                if best_conf < conf:
                    best_letter, best_conf = 'w', min(0.95, conf)

            # --- X: Index hooked (semi-bent), others curled tightly ---
            if 0.52 < idx_r < 0.72 and thumb_ratio < 0.65:
                if all(r < 0.52 for r in [mid_r, ring_r, pinky_r]):
                    conf = 0.68
                    hook_quality = 1.0 - abs(idx_r - 0.62) / 0.10
                    conf += 0.12 * max(0.0, hook_quality)
                    conf += 0.05 * (1.0 - np.mean([mid_r, ring_r, pinky_r]) / 0.52)
                    if best_conf < conf:
                        best_letter, best_conf = 'x', min(0.95, conf)

            # --- O: All curved, ALL fingertips touch thumb ---
            if all(d < 0.15 for d in thumb_dists):
                avg_ratio = np.mean(ratios)
                if avg_ratio < 0.85:
                    conf = 0.72
                    avg_dist = np.mean(thumb_dists)
                    conf += 0.15 * max(0.0, (0.15 - avg_dist) / 0.15)
                    if best_conf < conf:
                        best_letter, best_conf = 'o', min(0.95, conf)

            # --- M: Index, Middle, Ring over thumb (pointing down/forward) ---
            if all(r > 0.50 for r in [idx_r, mid_r, ring_r]) and pinky_r < 0.80:
                thumb_pos = self._thumb_position_relative(coords)
                if thumb_pos == 'under' or coords[4][1] > coords[6][1]:
                    # M: thumb tip must be under or behind ring finger
                    d_thumb_ring_pip = np.linalg.norm(coords[4] - coords[14])
                    if d_thumb_ring_pip < 0.40:
                        conf = 0.68
                        conf += 0.08 * min(1.0, (np.mean([idx_r, mid_r, ring_r]) - 0.50) / 0.40)
                        if best_conf < conf:
                            best_letter, best_conf = 'm', min(0.95, conf)

            # --- N: Index, Middle over thumb, Ring & Pinky curled ---
            if all(r > 0.50 for r in [idx_r, mid_r]) and all(r < 0.70 for r in [ring_r, pinky_r]):
                thumb_pos = self._thumb_position_relative(coords)
                if thumb_pos == 'under' or coords[4][1] > coords[6][1]:
                    # N: thumb under index+middle, but NOT ring
                    d_thumb_ring_pip = np.linalg.norm(coords[4] - coords[14])
                    if d_thumb_ring_pip >= 0.25:
                        conf = 0.65
                        conf += 0.08 * min(1.0, (np.mean([idx_r, mid_r]) - 0.50) / 0.40)
                        if best_conf < conf:
                            best_letter, best_conf = 'n', min(0.95, conf)

            # --- T: All curled, thumb tucked BETWEEN index and middle ---
            if all(r < 0.70 for r in ratios):
                thumb_pos = self._thumb_position_relative(coords)
                if thumb_pos == 'between':
                    conf = 0.68
                    conf += 0.10 * (1.0 - np.mean(ratios) / 0.70)
                    # Check thumb tip is actually between index and middle MCPs
                    mid_mcp = (coords[5] + coords[9]) / 2.0
                    d_between = np.linalg.norm(coords[4] - mid_mcp)
                    d_gap = np.linalg.norm(coords[5] - coords[9])
                    if d_between < d_gap:
                        conf += 0.07
                    if best_conf < conf:
                        best_letter, best_conf = 't', min(0.95, conf)

        # ============================================================== #
        #  PALM SIDEWAYS  (rotation >= 50°)                                #
        # ============================================================== #
        if rotation_deg >= 50.0:

            # --- C: All arched (0.30-0.92), thumb-index distance moderate ---
            if all(0.30 <= r <= 0.92 for r in ratios):
                if 0.45 <= d_thumb_idx <= 1.50 or 0.45 <= d_thumb_mid <= 1.50:
                    avg_ratio = np.mean(ratios)
                    conf = 0.65
                    # Best when ratios are in the middle (arched, not straight or curled)
                    arch_quality = 1.0 - abs(avg_ratio - 0.60) / 0.30
                    conf += 0.15 * max(0.0, arch_quality)
                    # Bonus for good thumb separation
                    if 0.60 <= d_thumb_idx <= 1.20:
                        conf += 0.10
                    if best_conf < conf:
                        best_letter, best_conf = 'c', min(0.95, conf)

        # ============================================================== #
        #  ORIENTATION-INDEPENDENT (OR BASED ON POINTING ANGLE)           #
        # ============================================================== #

        # --- G: Index semi-extended, thumb extended parallel, hand horizontal ---
        if idx_r > 0.70 and thumb_ratio > 0.65:
            if all(r < 0.70 for r in [mid_r, ring_r, pinky_r]):
                if 50.0 <= pointing_up_angle <= 140.0:
                    conf = 0.65
                    conf += 0.08 * min(1.0, (idx_r - 0.70) / 0.30)
                    conf += 0.07 * min(1.0, (thumb_ratio - 0.65) / 0.35)
                    conf += 0.05 * (1.0 - abs(pointing_up_angle - 90.0) / 40.0)
                    if best_conf < conf:
                        best_letter, best_conf = 'g', min(0.95, conf)

        # --- H: Index + Middle extended horizontally, others curled ---
        if all(r > 0.80 for r in [idx_r, mid_r]):
            if all(r < 0.70 for r in [ring_r, pinky_r]):
                if 50.0 <= pointing_up_angle <= 140.0:
                    conf = 0.66
                    conf += 0.08 * min(1.0, (np.mean([idx_r, mid_r]) - 0.80) / 0.20)
                    conf += 0.06 * (1.0 - np.mean([ring_r, pinky_r]) / 0.70)
                    conf += 0.05 * (1.0 - abs(pointing_up_angle - 90.0) / 40.0)
                    if best_conf < conf:
                        best_letter, best_conf = 'h', min(0.95, conf)

        # --- P: K-shape but hand points downward ---
        if idx_r > 0.80 and 0.50 < mid_r < 0.85:
            if all(r < 0.70 for r in [ring_r, pinky_r]):
                if pointing_up_angle > 110.0:
                    conf = 0.64
                    conf += 0.08 * min(1.0, (idx_r - 0.80) / 0.20)
                    conf += 0.08 * min(1.0, (pointing_up_angle - 110.0) / 70.0)
                    if best_conf < conf:
                        best_letter, best_conf = 'p', min(0.95, conf)


        # ============================================================== #
        #  BORDER ZONE: Accept C even when slightly under 50°             #
        # ============================================================== #
        if 35.0 <= rotation_deg < 50.0:
            # C can appear with palm slightly turned
            if all(0.30 <= r <= 0.92 for r in ratios):
                if 0.45 <= d_thumb_idx <= 1.50 or 0.45 <= d_thumb_mid <= 1.50:
                    avg_ratio = np.mean(ratios)
                    arch_quality = 1.0 - abs(avg_ratio - 0.60) / 0.30
                    conf = 0.58 + 0.12 * max(0.0, arch_quality)
                    if 0.60 <= d_thumb_idx <= 1.20:
                        conf += 0.08
                    if best_conf < conf:
                        best_letter, best_conf = 'c', min(0.90, conf)

        return best_letter, best_conf

    # ------------------------------------------------------------------ #
    #                         SANITY CHECKS                               #
    # ------------------------------------------------------------------ #

    def _apply_sanity_checks(self, pred, conf, features):
        """
        Post-hoc sanity checks on ML predictions to catch confusable pairs.
        Covers: B↔C, A↔E↔S, D↔F, U↔V, K↔V, I↔Y, M↔N↔T, R↔U, L↔D,
                W↔B, X↔D, O↔E, G↔H, P↔K.
        """
        ratios, thumb_ratio, coords = self._get_finger_ratios(features)
        if ratios is None:
            return pred, conf

        idx_r, mid_r, ring_r, pinky_r = ratios
        num_straight = sum(1 for r in ratios if r > 0.85)
        num_curled = sum(1 for r in ratios if r < 0.78)
        mean_ratio = np.mean(ratios)

        rotation_deg = self._hand_rotation_degrees(coords)
        alignment = self._palm_finger_alignment(coords)
        thumb_dists = self._thumb_to_fingertip_dists(coords)
        d_thumb_idx, d_thumb_mid, d_thumb_ring, d_thumb_pinky = thumb_dists
        angles = self._inter_finger_angles(coords)
        a_thumb_idx, a_idx_mid, a_mid_ring, a_ring_pinky = angles
        d_idx_mid_tips = self._finger_tip_distance(coords, 8, 12)

        # ---- B ↔ C ----
        if pred == 'b':
            if mean_ratio < 0.78 or num_straight < 3 or thumb_ratio >= 0.88:
                # Fingers not straight enough for B
                if all(0.30 <= r <= 0.92 for r in ratios) and rotation_deg >= 35:
                    if 0.45 <= d_thumb_idx <= 1.50:
                        return 'c', conf * 0.90
                elif mean_ratio < 0.75 and num_curled >= 3:
                    return ('e' if thumb_ratio < 0.68 else 'a'), conf * 0.85
                return pred, conf * 0.70

        elif pred == 'c':
            if mean_ratio > 0.92 or num_straight == 4:
                if alignment > 0.82 and thumb_ratio < 0.82 and rotation_deg < 50:
                    return 'b', conf * 0.85
            if mean_ratio < 0.30:
                return ('e' if thumb_ratio < 0.68 else 'a'), conf * 0.80

        # ---- A ↔ E ↔ S ----
        elif pred in ['a', 'e', 's']:
            if mean_ratio > 0.78 or num_straight >= 3:
                # Fingers not curled enough for A/E/S
                if mean_ratio >= 0.82 and num_straight >= 3:
                    return 'b', conf * 0.80
                return pred, conf * 0.60
            
            if pred == 'a':
                if thumb_ratio < 0.60:
                    return 'e', conf * 0.90  # Thumb strongly curled → E not A
                thumb_pos = self._thumb_position_relative(coords)
                if thumb_pos == 'front' and all(r < 0.65 for r in ratios):
                    return 's', conf * 0.85  # Thumb in front + tight fist → S
            
            elif pred == 'e':
                if thumb_ratio >= 0.80:
                    return 'a', conf * 0.90  # Thumb clearly extended → A not E
                if all(r < 0.65 for r in ratios):
                    thumb_pos = self._thumb_position_relative(coords)
                    if thumb_pos == 'front':
                        return 's', conf * 0.85

            elif pred == 's':
                if not all(r < 0.75 for r in ratios):
                    # Not tight enough for S
                    if thumb_ratio < 0.65:
                        return 'e', conf * 0.85
                    else:
                        return 'a', conf * 0.85

        # ---- D ↔ F ----
        elif pred == 'd':
            if idx_r < 0.70:
                # Index not extended — can't be D
                if all(r > 0.85 for r in [mid_r, ring_r, pinky_r]) and d_thumb_idx < 0.18:
                    return 'f', conf * 0.80
                if mean_ratio >= 0.82 and num_straight >= 3:
                    return 'b', conf * 0.75
                if mean_ratio < 0.75 and num_curled >= 3:
                    return ('e' if thumb_ratio < 0.68 else 'a'), conf * 0.80
                return pred, conf * 0.60

        elif pred == 'f':
            if not all(r > 0.85 for r in [mid_r, ring_r, pinky_r]):
                return pred, conf * 0.60
            if d_thumb_idx >= 0.25:
                # Thumb not touching index tip — maybe D or W
                if idx_r > 0.85 and all(r < 0.70 for r in [mid_r, ring_r, pinky_r]):
                    return 'd', conf * 0.75
                return pred, conf * 0.65

        # ---- U ↔ V ↔ R ----
        elif pred == 'u':
            # Check for R (crossed fingers) before dropping confidence due to bent fingers
            # In U, fingers are parallel and tips are together. 
            # In R, fingers are crossed so the angle is tight but tips are vertically separated.
            if d_idx_mid_tips > 0.12 and a_idx_mid < 0.18:
                return 'r', conf * 0.95  # Fingers crossed → R
            if a_idx_mid > 0.20:
                return 'v', conf * 0.90  # Fingers spread → V
            if not all(r > 0.85 for r in [idx_r, mid_r]):
                return pred, conf * 0.60

        elif pred == 'v':
            if not all(r > 0.85 for r in [idx_r, mid_r]):
                return pred, conf * 0.60
            if a_idx_mid < 0.12:
                return 'u', conf * 0.90  # Fingers together → U
            if d_idx_mid_tips < 0.08 and a_idx_mid < 0.10:
                return 'r', conf * 0.85

        elif pred == 'r':
            if not all(r > 0.70 for r in [idx_r, mid_r]):
                return pred, conf * 0.60
            if d_idx_mid_tips < 0.10 and a_idx_mid < 0.15:
                return 'u', conf * 0.90  # Tips together -> U
            if d_idx_mid_tips >= 0.25 and a_idx_mid > 0.20:
                return 'v', conf * 0.85

        # ---- K ↔ V ↔ P ----
        elif pred == 'k':
            pointing_up_angle = self._hand_pointing_angle(coords)
            if pointing_up_angle > 90.0:
                return 'p', conf * 0.85
            if not (idx_r > 0.80 and 0.40 < mid_r < 0.90):
                if all(r > 0.85 for r in [idx_r, mid_r]) and a_idx_mid > 0.20:
                    return 'v', conf * 0.80
                return pred, conf * 0.60
                
        elif pred == 'p':
            pointing_up_angle = self._hand_pointing_angle(coords)
            if pointing_up_angle < 60.0:
                return 'k', conf * 0.85

        # ---- I ↔ Y ----
        elif pred == 'i':
            if pinky_r < 0.80:
                return pred, conf * 0.55
            if thumb_ratio > 0.80 and all(r < 0.65 for r in [idx_r, mid_r, ring_r]):
                return 'y', conf * 0.90  # Thumb also extended → Y

        elif pred == 'y':
            if pinky_r < 0.75 or thumb_ratio < 0.65:
                if pinky_r > 0.75 and thumb_ratio < 0.65:
                    return 'i', conf * 0.85
                return pred, conf * 0.55

        # ---- M ↔ N ↔ T ----
        elif pred in ['m', 'n', 't']:
            thumb_pos = self._thumb_position_relative(coords)
            if pred == 'm' and thumb_pos == 'between':
                return 't', conf * 0.85
            if pred == 't' and thumb_pos == 'under':
                d_thumb_ring_pip = np.linalg.norm(coords[4] - coords[14])
                if d_thumb_ring_pip < 0.40:
                    return 'm', conf * 0.80
                else:
                    return 'n', conf * 0.80
            if pred == 'n' and thumb_pos == 'between':
                return 't', conf * 0.85

        # ---- L ↔ D ----
        elif pred == 'l':
            if not (idx_r > 0.85 and thumb_ratio > 0.80):
                if idx_r > 0.85 and d_thumb_mid < 0.25:
                    return 'd', conf * 0.80
                return pred, conf * 0.60

        # ---- W ↔ B ----
        elif pred == 'w':
            if pinky_r > 0.85:
                # Pinky also straight → probably B
                if alignment > 0.78 and thumb_ratio < 0.88:
                    return 'b', conf * 0.85
                return pred, conf * 0.70

        # ---- X ↔ D ----
        elif pred == 'x':
            if idx_r > 0.85:
                # Index too straight for a hook → D?
                if all(r < 0.70 for r in [mid_r, ring_r, pinky_r]) and d_thumb_mid < 0.25:
                    return 'd', conf * 0.80
                return pred, conf * 0.65
            if idx_r < 0.45:
                # Index too curled for a hook → fist letter
                if all(r < 0.65 for r in ratios):
                    return ('e' if thumb_ratio < 0.68 else 'a'), conf * 0.75
                return pred, conf * 0.60

        # ---- O ↔ E ----
        elif pred == 'o':
            # O requires all fingertips touching thumb
            if not all(d < 0.20 for d in thumb_dists):
                if all(r < 0.78 for r in ratios) and thumb_ratio < 0.68:
                    return 'e', conf * 0.80
                return pred, conf * 0.60

        # ---- G ↔ H ----
        elif pred == 'g':
            if all(r > 0.80 for r in [idx_r, mid_r]):
                return 'h', conf * 0.85

        elif pred == 'h':
            if mid_r < 0.70:
                return 'g', conf * 0.85

        # ---- P ↔ K ----
        elif pred == 'p':
            finger_orient_y = self._finger_orientation_y(coords)
            if finger_orient_y < 0.30:
                # Fingers not pointing down → K
                return 'k', conf * 0.80

        return pred, conf

    # ------------------------------------------------------------------ #
    #                    STATIC LETTER PREDICTION (MERGER)                #
    # ------------------------------------------------------------------ #

    def predict_static_letter(self, features):
        """
        Runs single-frame classification on static letters using a
        confidence-gated merger of rule-based and ML predictions.
        """
        features_static = features.copy()
        features_static[0:3] = 0.0
        features_static[63:66] = 0.0

        # 1. Rule-based prediction
        rule_pred, rule_conf = self.classify_by_rules(features_static)

        # FAST EXIT to eliminate lag: if rules are highly confident, skip heavy ML
        if rule_pred and rule_conf >= 0.80:
            return rule_pred, rule_conf

        # 2. ML prediction
        if self.static_classifier is not None:
            features_rich = utils.engineer_static_features(features_static.reshape(1, -1)).reshape(1, -1)
            prob = self.static_classifier.predict_proba(features_rich)[0]
            max_idx = np.argmax(prob)
            ml_pred = self.static_classes[max_idx]
            ml_conf = prob[max_idx]
        else:
            ml_pred, ml_conf = None, 0.0

        # 3. Confidence-gated merger
        # Case B: Strong ML prediction
        if ml_pred and ml_conf >= 0.70:
            final_pred, final_conf = self._apply_sanity_checks(ml_pred, ml_conf, features_static)
            if rule_pred == final_pred:
                final_conf = min(0.99, final_conf + 0.10)
            return final_pred, final_conf

        # Case C: Moderate rule prediction
        elif rule_pred:
            if ml_pred and rule_pred == ml_pred:
                return rule_pred, min(0.99, rule_conf + 0.10)
            return rule_pred, rule_conf * 0.85

        # Case D: Weak ML prediction
        elif ml_pred:
            return self._apply_sanity_checks(ml_pred, ml_conf, features_static)

        return None, 0.0

    # ------------------------------------------------------------------ #
    #                    DYNAMIC / LSTM PREDICTION                        #
    # ------------------------------------------------------------------ #

    def _predict_lstm_generic(self, model, classes, min_len=15, threshold=0.8):
        if model is None or len(self.sequence) < min_len:
            return None, None, 0.0
            
        seq_array = np.array(self.sequence, dtype=np.float32)
        seq_array = utils.normalize_sequence_wrists(seq_array)
        # Pad sequence with zeros if it hasn't reached full length of 30
        if len(self.sequence) < 30:
            padded = np.zeros((30, 126), dtype=np.float32)
            padded[-len(self.sequence):] = seq_array
            seq_array = padded
            
        seq_tensor = torch.tensor(seq_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = model(seq_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            max_prob, max_idx = torch.max(probabilities, dim=0)
            
        prediction = classes[max_idx.item()]
        confidence = max_prob.item()
        
        if prediction == 'neutral':
            return None, None, 0.0
            
        if confidence > threshold:
            return prediction, prediction, confidence
        return None, prediction, confidence

    def predict_dynamic_letter(self):
        """
        Predicts letters that require hand movement (J, K, Ñ, Q, Z, etc.)
        """
        return self._predict_lstm_generic(self.dynamic_letters_model, self.dynamic_letters_classes, min_len=15, threshold=0.50)

    def predict_word(self):
        """
        Predicts full gesture words (hola, gracias, etc.)
        """
        return self._predict_lstm_generic(self.words_model, self.words_classes, min_len=20, threshold=0.6)

    def process_frame(self, features, mode='static'):
        """
        Accepts raw 126 features and outputs translated letter or word.
        """
        # A. Independent per-hand carry-over for occlusion handling
        if not np.all(features == 0):
            left_hand = features[0:63]
            right_hand = features[63:126]
            
            # Left hand carry-over
            if np.all(left_hand == 0):
                self.left_lost_frames += 1
                if self.left_lost_frames <= 15 and self.last_valid_left is not None:
                    left_hand = self.last_valid_left
            else:
                self.left_lost_frames = 0
                self.last_valid_left = left_hand
                
            # Right hand carry-over
            if np.all(right_hand == 0):
                self.right_lost_frames += 1
                if self.right_lost_frames <= 15 and self.last_valid_right is not None:
                    right_hand = self.last_valid_right
            else:
                self.right_lost_frames = 0
                self.last_valid_right = right_hand
                
            features = np.concatenate([left_hand, right_hand])
            
        # 1. Handle hand-removal reset and carry-over dropouts
        if np.all(features == 0):
            self.no_hand_counter += 1
            if self.no_hand_counter > 30: # 1 second hand-absence timeout
                self.sequence.clear()
                self.last_committed_letter = None
                self.last_valid_left = None
                self.last_valid_right = None
            elif len(self.sequence) > 0 and self.no_hand_counter <= 10:
                # Carry over last valid hand features to bridge temporary tracking flickers
                self.sequence.append(self.sequence[-1])
            return None, None, 0.0
            
        self.no_hand_counter = 0
        self.sequence.append(features)
        self.frame_counter += 1
        
        # 2. Words Mode (translates full dynamic words)
        if mode == 'dynamic':
            if self.dynamic_cooldown > 0:
                self.dynamic_cooldown -= 1
                return None, None, 0.0
                
            # Decimation: only run LSTM every 5 frames to eliminate lag
            if self.frame_counter % 5 == 0:
                self.last_lstm_word_pred = self.predict_word()
                
            committed_word, raw_word, confidence = self.last_lstm_word_pred
            
            if committed_word is not None:
                self.dynamic_cooldown = 45 # prevent double-firing
                self.last_lstm_word_pred = (None, None, 0.0) # reset
                return committed_word, raw_word, confidence
            return None, raw_word, confidence
            
        # 3. Letters Mode (translates static & dynamic letters combined)
        # Calculate wrist standard deviation over last sequence to detect motion
        is_moving = False
        if len(self.sequence) >= 12:
            wrist_coords = []
            for seq_feat in self.sequence:
                left_wrist = seq_feat[0:3]
                right_wrist = seq_feat[63:66]
                if not np.all(left_wrist == 0):
                    wrist_coords.append(left_wrist)
                elif not np.all(right_wrist == 0):
                    wrist_coords.append(right_wrist)
            if len(wrist_coords) >= 10:
                stds = np.std(np.array(wrist_coords), axis=0)
                is_moving = np.max(stds) > 0.015 # motion threshold
                
        pred_letter, raw_letter, confidence = None, None, 0.0
        
        # Try dynamic letters first if motion is detected
        if is_moving:
            if self.frame_counter % 5 == 0:
                self.last_lstm_dyn_pred = self.predict_dynamic_letter()
            pred_letter, raw_letter, confidence = self.last_lstm_dyn_pred
            
        # Fall back to static letter classifier if still or dynamic confidence is low
        if pred_letter is None:
            static_pred, static_conf = self.predict_static_letter(features)
            if static_pred is not None:
                raw_letter = static_pred
                confidence = static_conf
                if static_conf > 0.65: # static letter threshold
                    pred_letter = static_pred
            
        if pred_letter is None:
            return None, raw_letter, confidence
            
        # Debouncing/smoothing loop (Letters require stable prediction over time)
        self.letter_history.append(pred_letter)
        
        if len(self.letter_history) >= 8 and len(set(list(self.letter_history)[-8:])) == 1:
            stable_letter = pred_letter
            if stable_letter != self.last_committed_letter:
                self.last_committed_letter = stable_letter
                return stable_letter, raw_letter, confidence
                
        return None, raw_letter, confidence
