import os
import sys

def check_kaggle_auth():
    # Check if kaggle.json exists in ~/.kaggle/
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("\n" + "="*60)
        print("ERROR: Kaggle API credentials not found.")
        print("="*60)
        print("To download the datasets automatically, you need a Kaggle account.")
        print("1. Go to https://www.kaggle.com/ and log in.")
        print("2. Go to your Account Settings (click your profile picture -> Settings).")
        print("3. Scroll down to 'API' and click 'Create New Token'.")
        print("4. This will download a 'kaggle.json' file.")
        print(f"5. Create a folder named '.kaggle' in {os.path.expanduser('~')} if it doesn't exist.")
        print(f"6. Place the 'kaggle.json' file inside {kaggle_dir}.")
        print("7. Run this script again.")
        print("="*60 + "\n")
        return False
    return True

def download_datasets():
    if not check_kaggle_auth():
        sys.exit(1)
        
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle library...")
        os.system("pip install kaggle")
        import kaggle
        
    kaggle.api.authenticate()
    
    print("\n--- Downloading Dataset 1: 100k Photos (Static Letters W, X, Q) ---")
    try:
        kaggle.api.dataset_download_files(
            'danieldiaz/lengua-de-senas-mexicana', 
            path='./data/kaggle_100k', 
            unzip=True
        )
        print("Dataset 1 downloaded successfully!")
    except Exception as e:
        print(f"Failed to download Dataset 1: {e}")

    print("\n--- Downloading Dataset 2: 249 Words (Dynamic Letters J, N, Z) ---")
    try:
        # NOTE: If we find the exact kaggle dataset name for the 249 words, we put it here.
        # As an example, we use a placeholder or search for it.
        # kaggle.api.dataset_download_files('username/datasetname', path='./data/kaggle_249', unzip=True)
        print("Note: The 249-word dataset URL needs to be explicitly provided. Skipping for now.")
    except Exception as e:
        print(f"Failed to download Dataset 2: {e}")

if __name__ == "__main__":
    download_datasets()
