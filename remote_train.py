import os
import shutil
import subprocess
import sys

print("=== 1. Fixing Remote Folder Paths ===")
target_data = '/content/LSM/data'
src_data = None

# Walk content directory to find where 'data' folder is
for root, dirs, files in os.walk('/content'):
    if 'drive' in root:
        continue
    if root == '/content/LSM':
        continue
    for d in dirs:
        if d == 'data':
            potential_src = os.path.join(root, d)
            # Count npy files in this data folder
            npy_count = sum(len(files) for r, ds, files in os.walk(potential_src) if any(f.endswith('.npy') for f in files))
            if npy_count > 0:
                src_data = potential_src
                print(f"Found valid data folder at {src_data} with {npy_count} files.")
                break
    if src_data:
        break

if src_data and src_data != target_data:
    if os.path.exists(target_data):
        print(f"Target path {target_data} already exists. Removing it first.")
        shutil.rmtree(target_data)
    print(f"Moving {src_data} to {target_data}...")
    shutil.move(src_data, target_data)
    print("Move complete!")
else:
    print("No valid nested data folder found to move, or it is already in the correct place.")

print("\n=== 2. Running train_models.py ===")
os.chdir('/content/LSM')
p = subprocess.run([sys.executable, 'train_models.py'], capture_output=True, text=True)
print("STDOUT:")
print(p.stdout)
print("STDERR:")
print(p.stderr)

print("\n=== 3. Exporting LSTM Models to ONNX ===")
import torch
import torch.onnx
import json
from train_models import LSMLSTMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def export_onnx(model_name, classes_name, output_onnx_name):
    model_path = os.path.join("models", model_name)
    classes_path = os.path.join("models", classes_name)
    output_path = os.path.join("models", output_onnx_name)
    
    if not os.path.exists(model_path):
        print(f"[Error] Model file {model_path} not found. Skipping export.")
        return
        
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    num_classes = len(classes)
    
    # Initialize model
    model = LSMLSTMModel(input_dim=126, hidden_dim=64, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 30, 126).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"[ONNX] Successfully exported {model_name} -> {output_onnx_name}")

export_onnx("dynamic_letters_model.pth", "dynamic_letters_classes.json", "dynamic_letters_model.onnx")
export_onnx("words_model.pth", "words_classes.json", "words_model.onnx")

print("\n=== Remote Job Complete ===")
