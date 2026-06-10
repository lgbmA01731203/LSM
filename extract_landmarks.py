"""
extract_landmarks.py
====================
Extracts MediaPipe hand landmarks from MSL-ABC archive images and saves
them as .npy files for training the static letters classifier.

Archives:
  - archive (3)/MSL-ABC/lsm-abc-{A,B,C}/{train,test}/{0..20}/*.jpg

Class index → LSM letter mapping (21 classes):
  0→a, 1→b, 2→c, 3→d, 4→e, 5→f, 6→g, 7→h, 8→i, 9→k,
  10→l, 11→m, 12→n, 13→o, 14→p, 15→q, 16→r, 17→s, 18→t,
  19→u, 20→v

Note: J is excluded (dynamic letter), Ñ is excluded.
      W, X, Y, Z are handled by rules only (no archive data).
      Q and K appear in archives but are also trained as dynamic letters.

Usage:
  python extract_landmarks.py [--max-per-class 500] [--archive-dir "archive (3)"]
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import cv2

# --- Index-to-letter mapping for MSL-ABC archives ---
INDEX_TO_LETTER = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e',
    5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'l',
    10: 'm', 11: 'n', 12: 'o', 13: 'p', 14: 'r',
    15: 's', 16: 't', 17: 'u', 18: 'v', 19: 'w', 20: 'y'
}

# Only extract for static letters we want to train with ML
# (f, g, h, i, k, l, m, n, o, p, r, s, t, u, v are NEW from archives)
# (a, b, c, d, e already have webcam data but archive data will supplement)
STATIC_LETTERS_TO_EXTRACT = set(INDEX_TO_LETTER.values())

DATA_OUTPUT_DIR = os.path.join('data', 'static_letters')


def setup_detector():
    """Initialize the GPU hand detector (same one used by the app)."""
    from gpu_hand_detector import GPUHandDetector
    detector = GPUHandDetector(
        palm_model_path='palm_detection.onnx',
        hand_model_path='handpose_estimation_mediapipe_2023feb.onnx'
    )
    if not detector.load():
        print("[ERROR] Could not load GPU hand detector. Ensure ONNX models are present.")
        sys.exit(1)
    return detector


def extract_features_from_image(detector, image_path):
    """
    Run hand detection on a single image and return 126-dim feature vector.
    Returns None if no hand is detected.
    """
    import utils
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Run detection
    result = detector.detect(img)
    
    if not result or not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None
    
    # Extract 126-dim features (same pipeline as live camera)
    features = utils.extract_features(result)
    
    # Verify it's not all zeros (no valid hand)
    if np.all(features == 0):
        return None
    
    return features


def collect_image_paths(archive_dir, class_idx):
    """
    Collect all image paths for a given class index across all archive parts and splits.
    Looks in: {archive_dir}/MSL-ABC/lsm-abc-{A,B,C}/{train,test}/{class_idx}/*.jpg
    """
    paths = []
    
    parts = ['lsm-abc-A', 'lsm-abc-B', 'lsm-abc-C']
    splits = ['train', 'test']
    
    for part in parts:
        for split in splits:
            class_dir = os.path.join(archive_dir, 'MSL-ABC', part, split, str(class_idx))
            if os.path.isdir(class_dir):
                jpgs = glob.glob(os.path.join(class_dir, '*.jpg'))
                paths.extend(jpgs)
    
    return paths


def extract_for_class(detector, archive_dir, class_idx, letter, max_per_class):
    """Extract landmarks for a single letter class."""
    # Collect all available image paths
    image_paths = collect_image_paths(archive_dir, class_idx)
    
    if not image_paths:
        print(f"  [{letter.upper()}] No images found for class index {class_idx}. Skipping.")
        return 0
    
    # Shuffle and cap at max_per_class
    random.seed(42 + class_idx)  # Reproducible sampling
    if len(image_paths) > max_per_class:
        image_paths = random.sample(image_paths, max_per_class)
    
    print(f"  [{letter.upper()}] Processing {len(image_paths)} images (of {len(collect_image_paths(archive_dir, class_idx))} available)...")
    
    # Create output directory
    output_dir = os.path.join(DATA_OUTPUT_DIR, letter)
    os.makedirs(output_dir, exist_ok=True)
    
    # Count existing archive-extracted samples to avoid overwriting
    existing = [f for f in os.listdir(output_dir) if f.startswith('archive_') and f.endswith('.npy')]
    start_idx = len(existing)
    
    extracted = 0
    failed = 0
    
    for i, img_path in enumerate(image_paths):
        features = extract_features_from_image(detector, img_path)
        if features is not None:
            filename = f"archive_{start_idx + extracted:04d}.npy"
            np.save(os.path.join(output_dir, filename), features)
            extracted += 1
        else:
            failed += 1
        
        # Progress update every 100 images
        if (i + 1) % 100 == 0:
            print(f"    ... processed {i+1}/{len(image_paths)} ({extracted} extracted, {failed} failed)")
    
    print(f"  [{letter.upper()}] Done: {extracted} landmarks extracted, {failed} images had no detectable hand.")
    return extracted


def main():
    parser = argparse.ArgumentParser(description='Extract hand landmarks from MSL-ABC archive images')
    parser.add_argument('--max-per-class', type=int, default=500,
                        help='Maximum number of images to process per class (default: 500)')
    parser.add_argument('--archive-dir', type=str, default='archive (3)',
                        help='Path to archive directory containing MSL-ABC (default: "archive (3)")')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MSL-ABC Archive Landmark Extractor")
    print(f"Archive: {args.archive_dir}")
    print(f"Max per class: {args.max_per_class}")
    print(f"Output: {DATA_OUTPUT_DIR}")
    print("=" * 60)
    
    # Verify archive exists
    msl_path = os.path.join(args.archive_dir, 'MSL-ABC')
    if not os.path.isdir(msl_path):
        print(f"[ERROR] Archive directory not found: {msl_path}")
        sys.exit(1)
    
    # Initialize detector
    print("\nInitializing hand detector...")
    detector = setup_detector()
    
    # Process each class
    total_extracted = 0
    print(f"\nExtracting landmarks for {len(INDEX_TO_LETTER)} letter classes...\n")
    
    for class_idx in sorted(INDEX_TO_LETTER.keys()):
        letter = INDEX_TO_LETTER[class_idx]
        if letter not in STATIC_LETTERS_TO_EXTRACT:
            continue
        
        count = extract_for_class(detector, args.archive_dir, class_idx, letter, args.max_per_class)
        total_extracted += count
    
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"Total landmarks extracted: {total_extracted}")
    print(f"Output directory: {os.path.abspath(DATA_OUTPUT_DIR)}")
    print(f"{'=' * 60}")
    
    detector.close()


if __name__ == '__main__':
    main()
