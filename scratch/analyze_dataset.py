import numpy as np
import os

def check_straightness(folder):
    if not os.path.exists(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')][:100]
    if not files:
        return None
    
    fingers = [
        # (Tip, DIP, PIP, MCP)
        (8, 7, 6, 5),     # Index
        (12, 11, 10, 9),  # Middle
        (16, 15, 14, 13), # Ring
        (20, 19, 18, 17)  # Pinky
    ]
    
    ratios = []
    for f in files:
        data = np.load(f)
        left = data[0:63]
        right = data[63:126]
        hand = left if not np.all(left == 0) else right
        if np.all(hand == 0):
            continue
        coords = hand.reshape(21, 3)
        
        finger_ratios = []
        for tip, dip, pip, mcp in fingers:
            d_mcp_tip = np.linalg.norm(coords[tip] - coords[mcp])
            sum_segments = (np.linalg.norm(coords[pip] - coords[mcp]) + 
                            np.linalg.norm(coords[dip] - coords[pip]) + 
                            np.linalg.norm(coords[tip] - coords[dip]))
            ratio = d_mcp_tip / sum_segments if sum_segments > 0 else 0
            finger_ratios.append(ratio)
        ratios.append(np.mean(finger_ratios))
    return np.mean(ratios)

print("A (folder a) average straightness ratio:", check_straightness('data/static_letters/a'))
print("B (folder b) average straightness ratio:", check_straightness('data/static_letters/b'))
print("C (folder c) average straightness ratio:", check_straightness('data/static_letters/c'))
