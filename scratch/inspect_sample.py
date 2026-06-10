import numpy as np
import os

def inspect_files(folder, num_files=3):
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')][:num_files]
    print(f"\n--- Inspecting {folder} ---")
    for f in files:
        data = np.load(f)
        left = data[0:63]
        right = data[63:126]
        hand = left if not np.all(left == 0) else right
        if np.all(hand == 0):
            print(f"  File {os.path.basename(f)}: No hand detected.")
            continue
        coords = hand.reshape(21, 3)
        print(f"  File: {os.path.basename(f)}")
        # Print Index finger joints
        mcp, pip, dip, tip = coords[5], coords[6], coords[7], coords[8]
        d_mcp_tip = np.linalg.norm(tip - mcp)
        sum_segments = np.linalg.norm(pip - mcp) + np.linalg.norm(dip - pip) + np.linalg.norm(tip - dip)
        ratio = d_mcp_tip / sum_segments if sum_segments > 0 else 0
        print(f"    Index MCP: {mcp.round(3)}, PIP: {pip.round(3)}, DIP: {dip.round(3)}, Tip: {tip.round(3)}")
        print(f"    Index Straightness Ratio: {ratio:.3f}")

inspect_files('data/static_letters/a')
inspect_files('data/static_letters/b')
inspect_files('data/static_letters/c')
