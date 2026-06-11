import os
import shutil
import re
import pandas as pd
import numpy as np

# Base paths
WORKSPACE_DIR = r"C:\Users\capa_\OneDrive\Desktop\LSM"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "LSM_Dataset")

DYNAMIC_PROFILE_SRC = os.path.join(WORKSPACE_DIR, "MSL-dynamic-signs-profile", "MSL dynamic-profile-signs")
DYNAMIC_FRONTAL_SRC = os.path.join(WORKSPACE_DIR, "MSL-dynamic-signs-frontal-view", "MSL-dynamic-signs", "train")
WORDS_SRC = os.path.join(WORKSPACE_DIR, "Mexican sign language dataset", "MSLwords1")
ARCHIVES_SRC = [
    os.path.join(WORKSPACE_DIR, "archive (2)", "MSL-ABC"),
    os.path.join(workspace_dir := WORKSPACE_DIR, "archive (3)", "MSL-ABC")
]

# Ensure output subdirs exist
os.makedirs(os.path.join(OUTPUT_DIR, "static_letters"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "dynamic_letters"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "words"), exist_ok=True)

# Helper to clean strings for directory names
def clean_name(name):
    if not isinstance(name, str):
        return "unknown"
    name = name.strip().lower()
    # Replace special characters, accents, and spaces
    name = re.sub(r"[áäâà]", "a", name)
    name = re.sub(r"[éëêè]", "e", name)
    name = re.sub(r"[íïîì]", "i", name)
    name = re.sub(r"[óöôò]", "o", name)
    name = re.sub(r"[úüûù]", "u", name)
    name = re.sub(r"[ñ]", "n", name)
    # Alphanumeric and underscores only
    name = re.sub(r"[^a-z0-9_.-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def copy_file_if_changed(src, dest):
    """Copies file if it doesn't exist at dest or size differs, returning True if copied, False if skipped."""
    if os.path.exists(dest):
        try:
            if os.path.getsize(dest) == os.path.getsize(src):
                return False  # Skip
        except Exception:
            pass
    shutil.copy2(src, dest)
    return True

# 1. PROCESS DYNAMIC LETTERS
print("--- Processing Dynamic Letters ---")
dynamic_letters = ['j', 'k', 'q', 'x', 'z', 'ñ']
for letter in dynamic_letters:
    os.makedirs(os.path.join(OUTPUT_DIR, "dynamic_letters", letter), exist_ok=True)

# Frontal view
copied_frontal = 0
skipped_frontal = 0
if os.path.exists(DYNAMIC_FRONTAL_SRC):
    frontal_files = [f for f in os.listdir(DYNAMIC_FRONTAL_SRC) if f.lower().endswith(('.mp4', '.avi'))]
    for file in frontal_files:
        parts = file.split("-")
        if len(parts) >= 2:
            cls_raw = parts[1].lower()
            cls = cls_raw if cls_raw in ['j', 'k', 'q', 'x', 'z'] else 'ñ'
            src_file = os.path.join(DYNAMIC_FRONTAL_SRC, file)
            dest_file = os.path.join(OUTPUT_DIR, "dynamic_letters", cls, f"frontal_{file}")
            if copy_file_if_changed(src_file, dest_file):
                copied_frontal += 1
            else:
                skipped_frontal += 1
    print(f"Copied {copied_frontal} frontal dynamic videos (skipped {skipped_frontal} unchanged).")
else:
    print("Frontal dynamic source path not found.")

# Profile view
copied_profile = 0
skipped_profile = 0
if os.path.exists(DYNAMIC_PROFILE_SRC):
    subdirs = [d for d in os.listdir(DYNAMIC_PROFILE_SRC) if os.path.isdir(os.path.join(DYNAMIC_PROFILE_SRC, d))]
    for sub in subdirs:
        cls_raw = sub.lower()
        cls = cls_raw if cls_raw in ['j', 'k', 'q', 'x', 'z'] else 'ñ'
        sub_path = os.path.join(DYNAMIC_PROFILE_SRC, sub)
        videos = [f for f in os.listdir(sub_path) if f.lower().endswith(('.mp4', '.avi'))]
        for video in videos:
            src_file = os.path.join(sub_path, video)
            dest_file = os.path.join(OUTPUT_DIR, "dynamic_letters", cls, f"profile_{sub}_{video}")
            if copy_file_if_changed(src_file, dest_file):
                copied_profile += 1
            else:
                skipped_profile += 1
    print(f"Copied {copied_profile} profile dynamic videos (skipped {skipped_profile} unchanged).")
else:
    print("Profile dynamic source path not found.")


# 2. PROCESS WORDS
print("\n--- Processing Words ---")
excel_path = os.path.join(WORDS_SRC, "classes.xlsx")
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    
    word_mappings = {}
    for idx, row in df.iterrows():
        class_num = row['Class number']
        if pd.isna(class_num):
            continue
        word_raw = row['Word']
        if pd.isna(word_raw):
            continue
        class_num = int(class_num)
        word_name = clean_name(str(word_raw))
        word_mappings[class_num] = word_name

    copied_words = 0
    skipped_words = 0
    skipped_folders = []
    
    for class_num, word_name in word_mappings.items():
        folder_name = f"{class_num:03d}"
        folder_path = os.path.join(WORDS_SRC, folder_name)
        if os.path.exists(folder_path):
            dest_word_dir = os.path.join(OUTPUT_DIR, "words", word_name)
            os.makedirs(dest_word_dir, exist_ok=True)
            
            # Recursively walk folder_path to find all .mp4 and .mov videos
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi')):
                        src_file = os.path.join(root, file)
                        # Construct a unique file name based on subfolder structure
                        rel_path = os.path.relpath(src_file, folder_path)
                        new_name = rel_path.replace(os.sep, "_")
                        dest_file = os.path.join(dest_word_dir, f"{folder_name}_{new_name}")
                        
                        if copy_file_if_changed(src_file, dest_file):
                            copied_words += 1
                        else:
                            skipped_words += 1
        else:
            skipped_folders.append(folder_name)
            
    print(f"Copied {copied_words} word videos (skipped {skipped_words} unchanged) across {len(word_mappings) - len(skipped_folders)} classes.")
    if skipped_folders:
        print(f"Skipped {len(skipped_folders)} folders (not found on disk): {skipped_folders[:10]}...")
else:
    print("Excel mapping file for words not found.")


# 3. PROCESS STATIC LETTERS
print("\n--- Processing Static Letters ---")
static_letter_mapping = {
    '0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e',
    '5': 'f', '6': 'g', '7': 'h', '8': 'i', '9': 'l',
    '10': 'm', '11': 'n', '12': 'o', '13': 'p', '14': 'r',
    '15': 's', '16': 't', '17': 'u', '18': 'v', '19': 'w', '20': 'y'
}

# Create output folders for static letters
for letter in static_letter_mapping.values():
    os.makedirs(os.path.join(OUTPUT_DIR, "static_letters", letter), exist_ok=True)

copied_static = 0
skipped_static_mismatched = 0
skipped_static_unchanged = 0

for arch_idx, arch_path in enumerate(ARCHIVES_SRC, start=2):
    if not os.path.exists(arch_path):
        print(f"Archive source {arch_path} not found.")
        continue
        
    print(f"Scanning archive ({arch_idx}) at {arch_path}...")
    parts = [d for d in os.listdir(arch_path) if os.path.isdir(os.path.join(arch_path, d))]
    
    for part in parts:
        part_path = os.path.join(arch_path, part)
        splits = [d for d in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, d))]
        
        for split in splits:
            split_path = os.path.join(part_path, split)
            num_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            
            for num_dir in num_dirs:
                if num_dir not in static_letter_mapping:
                    continue
                letter = static_letter_mapping[num_dir]
                num_dir_path = os.path.join(split_path, num_dir)
                
                images = [f for f in os.listdir(num_dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                dest_dir = os.path.join(OUTPUT_DIR, "static_letters", letter)
                
                for img in images:
                    # Parse image name using regex to make sure it matches the letter,
                    # e.g. S1-A-4-0.jpg -> A
                    match = re.match(r"^S\d+-([A-Za-z]+)-", img)
                    if match:
                        parsed_letter = match.group(1).lower()
                        # Verify the parsed letter matches mapped letter to prevent class pollution
                        if parsed_letter != letter:
                            skipped_static_mismatched += 1
                            continue
                            
                    src_file = os.path.join(num_dir_path, img)
                    # Unique filename
                    new_name = f"arch{arch_idx}_{part}_{split}_{num_dir}_{img}"
                    dest_file = os.path.join(dest_dir, new_name)
                    if copy_file_if_changed(src_file, dest_file):
                        copied_static += 1
                    else:
                        skipped_static_unchanged += 1

print(f"Copied {copied_static} static images (skipped {skipped_static_unchanged} unchanged).")
print(f"Skipped {skipped_static_mismatched} mismatched static images.")

print("\n=======================================================")
print("DATASET CONSOLIDATION COMPLETED SUCCESSFULLY")
print(f"Output folder: {OUTPUT_DIR}")
print("=======================================================")
