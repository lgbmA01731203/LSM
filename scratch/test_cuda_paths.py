import os
import sys

# Find site-packages directory
import site
site_dirs = site.getsitepackages()
print(f"Site packages: {site_dirs}")

# Search for nvidia directories and add their DLL containing folders to PATH
nvidia_dirs = []
for sd in site_dirs:
    nvidia_root = os.path.join(sd, "nvidia")
    if os.path.exists(nvidia_root):
        print(f"Found nvidia root: {nvidia_root}")
        for root, dirs, files in os.walk(nvidia_root):
            # If there's a dll in this folder, add it to PATH
            if any(f.endswith(".dll") for f in files):
                nvidia_dirs.append(root)

print("Adding the following directories to PATH:")
for d in nvidia_dirs:
    print(f"  {d}")
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    # For Python 3.8+, we also need to use os.add_dll_directory
    try:
        os.add_dll_directory(d)
        print(f"  [OK] Added via add_dll_directory: {d}")
    except Exception as e:
        print(f"  [INFO] Could not add via add_dll_directory: {e}")

# Now import onnxruntime and check providers
try:
    import onnxruntime as ort
    print(f"Available providers: {ort.get_available_providers()}")
    session = ort.InferenceSession("palm_detection.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Active provider: {session.get_providers()}")
except Exception as e:
    print(f"Error during ORT initialization: {e}")
