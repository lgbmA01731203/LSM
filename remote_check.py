import os

print("=== Checking Google Drive Mount ===")
drive_exists = os.path.exists('/content/drive')
print(f"Does '/content/drive' exist? {drive_exists}")

if drive_exists:
    mydrive_path = '/content/drive/MyDrive'
    mydrive_exists = os.path.exists(mydrive_path)
    print(f"Does '{mydrive_path}' exist? {mydrive_exists}")
    
    if mydrive_exists:
        files = os.listdir(mydrive_path)
        print(f"Total files/folders in MyDrive: {len(files)}")
        zip_exists = 'colab_training_data.zip' in files
        print(f"Is 'colab_training_data.zip' in MyDrive? {zip_exists}")
        
        full_zip_path = '/content/drive/MyDrive/colab_training_data.zip'
        print(f"Does file exist at '{full_zip_path}'? {os.path.exists(full_zip_path)}")
        if os.path.exists(full_zip_path):
            print(f"File size: {os.path.getsize(full_zip_path)} bytes")
else:
    print("Drive folder does not exist at /content/drive")
