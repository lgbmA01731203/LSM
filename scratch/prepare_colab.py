import os
import zipfile

def main():
    zip_name = "colab_upload.zip"
    print(f"=== Preparing {zip_name} for Google Colab ===")
    
    # Files to include in the zip root
    code_files = [
        "mp_palmdet.py",
        "mp_handpose.py",
        "gpu_hand_detector.py",
        "utils.py"
    ]
    
    # Exclude models from zip to keep upload size smaller, Colab can download them in seconds!
    
    dataset_dir = r"Mexican sign language dataset"
    script_path = os.path.join("scratch", "extract_lsm_dataset.py")
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Local dataset folder '{dataset_dir}' not found. Please verify the path.")
        return
        
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add python code files
        for f in code_files:
            if os.path.exists(f):
                print(f"Adding code file: {f}")
                zipf.write(f)
            else:
                print(f"[WARNING] Code file {f} missing!")
                
        # 2. Add extraction script
        if os.path.exists(script_path):
            print(f"Adding script: {script_path}")
            zipf.write(script_path, arcname="extract_lsm_dataset.py")
            
        # 3. Add dataset images and spreadsheet
        print("Adding dataset folder (this may take a minute as it contains thousands of images)...")
        file_count = 0
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep path relative to workspace
                arcname = os.path.relpath(file_path, os.path.dirname(dataset_dir))
                zipf.write(file_path, arcname=arcname)
                file_count += 1
                if file_count % 5000 == 0:
                    print(f"  Added {file_count} files so far...")
                    
    print(f"\n[SUCCESS] Created '{zip_name}' containing {file_count + len(code_files) + 1} files!")
    print("Now upload this zip file directly to your Google Colab session.")

if __name__ == '__main__':
    main()
