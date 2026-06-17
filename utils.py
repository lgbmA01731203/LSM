import numpy as np
import cv2

# Define hand connections index pairs
CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

def extract_landmarks(hand_landmarks):
    """
    Extracts x, y, z coordinates from a single hand landmarks object
    and normalizes them (wrist translation + scale normalization).
    Returns a 63-dimensional numpy array, preserving raw wrist coordinates at index 0.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # 1. Translate relative to wrist (landmark 0 is wrist)
    wrist = coords[0]
    translated = coords - wrist
    
    # 2. Scale normalize relative to the maximum distance from wrist
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        normalized = translated / max_dist
    else:
        normalized = translated
        
    # Preserve raw wrist coordinates at index 0 (indices 0, 1, 2 in flattened array)
    # This enables dynamic trajectory tracking while keeping fingers normalized.
    normalized[0] = wrist
    return normalized.flatten()

def normalize_sequence_wrists(sequence):
    """
    Given a sequence of shape (N, 126) where the wrist coordinates (0:3 and 63:66)
    are raw coordinates, replaces them with the displacement relative to the first
    frame in the sequence where the hand is detected.
    """
    normalized_seq = sequence.copy()
    
    # Find the first frame where left hand is present
    left_start_wrist = None
    for f in range(len(sequence)):
        left_hand = sequence[f, 0:63]
        if not np.all(left_hand == 0):
            left_start_wrist = sequence[f, 0:3].copy()
            break
            
    # Find the first frame where right hand is present
    right_start_wrist = None
    for f in range(len(sequence)):
        right_hand = sequence[f, 63:126]
        if not np.all(right_hand == 0):
            right_start_wrist = sequence[f, 63:66].copy()
            break
            
    # Calculate displacements
    for f in range(len(sequence)):
        # Left hand
        left_hand = normalized_seq[f, 0:63]
        if not np.all(left_hand == 0) and left_start_wrist is not None:
            normalized_seq[f, 0:3] = normalized_seq[f, 0:3] - left_start_wrist
        else:
            normalized_seq[f, 0:3] = 0.0
            
        # Right hand
        right_hand = normalized_seq[f, 63:126]
        if not np.all(right_hand == 0) and right_start_wrist is not None:
            normalized_seq[f, 63:66] = normalized_seq[f, 63:66] - right_start_wrist
        else:
            normalized_seq[f, 63:66] = 0.0
            
    return normalized_seq

def compute_motion_features(seq: np.ndarray) -> np.ndarray:
    """
    Converts a (T, 126) position sequence into a (T, 378) motion-aware sequence
    by appending per-frame velocity and acceleration channels.

    Channels per frame:
      [0:126]   — normalized positions  (same as input)
      [126:252] — velocity  = pos[t] - pos[t-1]  (0 for t=0)
      [252:378] — acceleration = vel[t] - vel[t-1] (0 for t=0,1)

    These extra channels give the LSTM direct access to HOW FAST and HOW THE
    SPEED CHANGES for every landmark — the primary discriminator between dynamic
    signs that share a similar hand shape (e.g. j vs. z, q vs. k).
    """
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    vel[1:]  = seq[1:] - seq[:-1]
    acc[2:]  = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=1).astype(np.float32)


class HandLandmarkStabilizer:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_landmarks = {}  # key: "Left" or "Right", value: np.ndarray of shape (21, 3)
        self.missing_counters = {"Left": 0, "Right": 0}
        
    def smooth(self, detection_result):
        """
        Smooths detection_result.hand_landmarks in-place using EMA,
        grouped by handedness.
        """
        if not detection_result or not detection_result.hand_landmarks:
            self.missing_counters["Left"] += 1
            self.missing_counters["Right"] += 1
            if self.missing_counters["Left"] > 10:
                self.prev_landmarks.pop("Left", None)
            if self.missing_counters["Right"] > 10:
                self.prev_landmarks.pop("Right", None)
            return detection_result
            
        current_hands = set()
        
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Identify hand label
            label = "Right"
            if i < len(detection_result.handedness) and len(detection_result.handedness[i]) > 0:
                cat = detection_result.handedness[i][0]
                if hasattr(cat, 'category_name') and cat.category_name:
                    label = cat.category_name
                elif hasattr(cat, 'display_name') and cat.display_name:
                    label = cat.display_name
                else:
                    label = str(cat)
            
            hand_key = "Left" if "Left" in label else "Right"
            current_hands.add(hand_key)
            self.missing_counters[hand_key] = 0
            
            # Extract coordinates as np.array
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
            
            if hand_key in self.prev_landmarks:
                # Apply EMA: smoothed = alpha * current + (1 - alpha) * previous
                smoothed = self.alpha * coords + (1 - self.alpha) * self.prev_landmarks[hand_key]
                self.prev_landmarks[hand_key] = smoothed
            else:
                self.prev_landmarks[hand_key] = coords
                smoothed = coords
                
            # Update the landmarks in detection_result in-place
            for j, lm in enumerate(hand_landmarks):
                lm.x = float(smoothed[j, 0])
                lm.y = float(smoothed[j, 1])
                lm.z = float(smoothed[j, 2])
                
        # Handle hands that were not detected in this frame
        for hand_key in ["Left", "Right"]:
            if hand_key not in current_hands:
                self.missing_counters[hand_key] += 1
                if self.missing_counters[hand_key] > 10:
                    self.prev_landmarks.pop(hand_key, None)
                    
        return detection_result

def extract_features(detection_result):
    """
    Extracts features for both Left and Right hands from MediaPipe HandLandmarker result.
    If a hand is missing, it is represented as 63 zeros.
    Returns a combined 126-dimensional numpy array.
    """
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    
    if detection_result and detection_result.hand_landmarks:
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Extract handedness label (Left / Right)
            # Default to Right if we can't parse handedness
            label = "Right"
            if i < len(detection_result.handedness) and len(detection_result.handedness[i]) > 0:
                cat = detection_result.handedness[i][0]
                # category_name or display_name holds "Left" or "Right"
                if hasattr(cat, 'category_name') and cat.category_name:
                    label = cat.category_name
                elif hasattr(cat, 'display_name') and cat.display_name:
                    label = cat.display_name
                else:
                    label = str(cat)
                    
            features = extract_landmarks(hand_landmarks)
            
            # Map Left/Right from camera space
            # Note: MediaPipe flips labels under some conditions, but we map strictly to their tags
            if "Left" in label:
                left_hand = features
            elif "Right" in label:
                right_hand = features
                
    return np.concatenate([left_hand, right_hand])

def get_hand_orientation(features):
    """
    Determines the hand orientation (arriba, abajo, izquierda, derecha) 
    based on the vector from wrist (landmark 0) to middle finger MCP (landmark 9).
    Returns a tuple of (left_orient, right_orient), each being one of 
    ['arriba', 'abajo', 'izquierda', 'derecha', None].
    """
    left_hand = features[0:63]
    right_hand = features[63:126]
    
    def determine_orient(hand_feats):
        if np.all(hand_feats == 0):
            return None
        # Landmark 9 x is at index 27, y is at index 28 (relative to wrist which is at 0,0,0)
        x_9 = hand_feats[27]
        y_9 = hand_feats[28]
        
        # Compare absolute values to find dominant direction
        if abs(x_9) > abs(y_9):
            return "izquierda" if x_9 < 0 else "derecha"
        else:
            return "arriba" if y_9 < 0 else "abajo"
            
    left_orient = determine_orient(left_hand)
    right_orient = determine_orient(right_hand)
    return left_orient, right_orient

def engineer_static_features(X_raw):
    """
    Computes hand-crafted geometric features (distances, palm direction, finger straightness,
    hand rotation angle, inter-finger angles, thumb distances, PIP curl angles)
    specifically for the static letters classifier to improve accuracy.
    Accepts 1D (126,) or 2D (N, 126) arrays and returns the enriched features.
    
    Features per hand (39 total):
      1. Fingertip-to-wrist distances (5)
      2. Fingertip-to-MCP distances (5)
      3. Adjacent fingertip distances (4)
      4. Palm orientation unit vector (3)
      4b. Palm normal unit vector (3)
      5. Finger straightness ratios (5)
      6. Hand rotation angle in degrees (1) — 0°=facing camera, 90°=sideways
      7. Inter-finger angles from wrist (4) — thumb-index, index-mid, mid-ring, ring-pinky
      8. Thumb-to-fingertip distances (4) — thumb tip to index/mid/ring/pinky tips
      9. PIP joint curl angles in radians (5) — bend angle at each finger's PIP joint
    """
    is_1d = X_raw.ndim == 1
    if is_1d:
        X_raw = X_raw.reshape(1, -1)
        
    N = X_raw.shape[0]
    FEATS_PER_HAND = 39
    engineered = []
    
    # Landmark indices for fingertips and MCP bases
    fingertips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    
    # Finger segments for straightness calculation (tip, dip, pip, base/mcp)
    finger_joints = [
        (4, 3, 2, 1),     # Thumb
        (8, 7, 6, 5),     # Index
        (12, 11, 10, 9),  # Middle
        (16, 15, 14, 13), # Ring
        (20, 19, 18, 17)  # Pinky
    ]
    
    # PIP joint triplets (proximal, pip, distal) for bend angle calculation
    pip_triplets = [
        (1, 2, 3),     # Thumb: CMC → MCP → IP
        (5, 6, 7),     # Index: MCP → PIP → DIP
        (9, 10, 11),   # Middle: MCP → PIP → DIP
        (13, 14, 15),  # Ring: MCP → PIP → DIP
        (17, 18, 19)   # Pinky: MCP → PIP → DIP
    ]
    
    for i in range(N):
        row = X_raw[i]
        left_hand = row[0:63].reshape(21, 3)
        right_hand = row[63:126].reshape(21, 3)
        
        feats = []
        for hand in [left_hand, right_hand]:
            if np.all(hand == 0):
                feats.extend([0.0] * FEATS_PER_HAND)
                continue
                
            # 1. Fingertip-to-wrist distances (5 features)
            for tip in fingertips:
                dist = np.linalg.norm(hand[tip] - hand[0])
                feats.append(dist)
                
            # 2. Fingertip-to-MCP distances (5 features)
            for tip, mcp in zip(fingertips, mcps):
                dist = np.linalg.norm(hand[tip] - hand[mcp])
                feats.append(dist)
                
            # 3. Adjacent fingertip distances (4 features)
            for j in range(4):
                dist = np.linalg.norm(hand[fingertips[j]] - hand[fingertips[j+1]])
                feats.append(dist)
                
            # 4. Palm orientation unit vector (3 features)
            orient_vec = hand[9] - hand[0]
            norm = np.linalg.norm(orient_vec)
            if norm > 0:
                orient_vec = orient_vec / norm
            feats.extend(orient_vec.tolist())
            
            # 4b. Palm normal unit vector (3 features)
            v1 = hand[5] - hand[0]
            v2 = hand[17] - hand[0]
            normal_vec = np.cross(v1, v2)
            norm_val = np.linalg.norm(normal_vec)
            if norm_val > 0:
                normal_vec = normal_vec / norm_val
            feats.extend(normal_vec.tolist())
            
            # 5. Finger straightness ratios (5 features)
            for tip, dip, pip, mcp in finger_joints:
                d_mcp_tip = np.linalg.norm(hand[tip] - hand[mcp])
                sum_segments = (np.linalg.norm(hand[pip] - hand[mcp]) + 
                                np.linalg.norm(hand[dip] - hand[pip]) + 
                                np.linalg.norm(hand[tip] - hand[dip]))
                ratio = d_mcp_tip / sum_segments if sum_segments > 0 else 0.0
                feats.append(ratio)
            
            # 6. Hand rotation angle in degrees (1 feature)
            # 0° = palm facing camera (front), 90° = palm facing sideways
            abs_normal_z = abs(normal_vec[2]) if norm_val > 0 else 0.0
            rotation_deg = np.degrees(np.arccos(np.clip(abs_normal_z, 0.0, 1.0)))
            feats.append(rotation_deg)
            
            # 7. Inter-finger angles from wrist (4 features)
            # Angles between adjacent finger vectors as seen from the wrist
            angle_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]  # thumb-idx, idx-mid, mid-ring, ring-pinky
            for t1, t2 in angle_pairs:
                va = hand[t1] - hand[0]
                vb = hand[t2] - hand[0]
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na > 0 and nb > 0:
                    cos_a = np.dot(va, vb) / (na * nb)
                    feats.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                else:
                    feats.append(0.0)
            
            # 8. Thumb-to-fingertip distances (4 features)
            # Critical for O (thumb touches all), F/D (thumb touches index), A vs E
            for tip in [8, 12, 16, 20]:
                feats.append(np.linalg.norm(hand[4] - hand[tip]))
            
            # 9. PIP joint curl angles in radians (5 features)
            # Angle at the PIP joint — small angle = very bent, large angle = straight
            for prox, pip_j, dist in pip_triplets:
                va = hand[prox] - hand[pip_j]
                vb = hand[dist] - hand[pip_j]
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na > 0 and nb > 0:
                    cos_a = np.dot(va, vb) / (na * nb)
                    feats.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                else:
                    feats.append(0.0)
            
        engineered.append(feats)
        
    engineered = np.array(engineered)
    X_rich = np.hstack([X_raw, engineered])
    
    if is_1d:
        return X_rich[0]
    return X_rich

def draw_landmarks(image, detection_result):
    """
    Draws custom hand skeleton connections on the image.
    Uses neon green for lines and blue/white for joint nodes.
    """
    if not detection_result or not detection_result.hand_landmarks:
        return image
        
    h, w, _ = image.shape
    
    for hand_landmarks in detection_result.hand_landmarks:
        # Convert landmarks to pixel coordinates
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))
            
        # Draw skeleton lines (Neon Green/Teal)
        for start_idx, end_idx in CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], (0, 255, 127), 2)
                
        # Draw joint nodes (Blue with white center)
        for pt in points:
            cv2.circle(image, pt, 5, (255, 120, 0), -1)
            cv2.circle(image, pt, 2, (255, 255, 255), -1)
            
    return image
