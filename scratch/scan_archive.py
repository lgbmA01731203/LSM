import os

DATASET_ROOT = r"C:\Users\capa_\OneDrive\Desktop\LSM\archive (3)\MSL-ABC"

for folder in os.listdir(DATASET_ROOT):
    folder_path = os.path.join(DATASET_ROOT, folder)
    if os.path.isdir(folder_path):
        classes_found = set()
        train_path = os.path.join(folder_path, "train")
        if os.path.exists(train_path):
            for sub in os.listdir(train_path):
                sub_path = os.path.join(train_path, sub)
                if os.path.isdir(sub_path):
                    for f in os.listdir(sub_path):
                        if f.lower().endswith('.jpg'):
                            parts = f.split('-')
                            if len(parts) >= 3:
                                classes_found.add(parts[2])
        print(f"Folder {folder} contains class indices: {classes_found}")
