"""
remote_train.py — Google Colab training script
===============================================
Instrucciones:
  1. Sube todo el repo a Google Drive (o clona desde GitHub)
  2. En Colab: Runtime → Change runtime type → T4 GPU
  3. Monta Drive:
       from google.colab import drive
       drive.mount('/content/drive')
  4. Copia el repo al entorno local de Colab (más rápido que leer desde Drive):
       !cp -r /content/drive/MyDrive/LSM /content/LSM
  5. Instala dependencias:
       !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
       !pip install numpy scikit-learn joblib
  6. Corre esta celda:
       %cd /content/LSM
       !python remote_train.py
  7. Copia los modelos de vuelta a Drive:
       !cp /content/LSM/models/*.pth /content/drive/MyDrive/LSM/models/
       !cp /content/LSM/models/*.json /content/drive/MyDrive/LSM/models/
"""

import os, sys, json, shutil, subprocess
import torch
import torch.onnx

print("=== LSM Remote Training (BiLSTM + Attention + Motion Features) ===")

# ---- 1. Fix paths: find data folder wherever Colab unzipped it ----
target_data = '/content/LSM/data'
src_data = None
for root, dirs, files in os.walk('/content'):
    if 'drive' in root:
        continue
    for d in dirs:
        if d == 'data':
            candidate = os.path.join(root, d)
            npy_count = sum(
                len([f for f in fs if f.endswith('.npy')])
                for _, _, fs in os.walk(candidate)
            )
            if npy_count > 0:
                src_data = candidate
                print(f"Found data at {src_data} ({npy_count} .npy files)")
                break
    if src_data:
        break

if src_data and src_data != target_data:
    if os.path.exists(target_data):
        shutil.rmtree(target_data)
    shutil.move(src_data, target_data)
    print(f"Moved data -> {target_data}")

# ---- 2. Train ----
os.chdir('/content/LSM')
print("\n=== Training Dynamic Letters ===")
from train_models import train_dynamic_letters, train_words, LSMLSTMModel, MOTION_INPUT_DIM

train_dynamic_letters()

print("\n=== Training Words ===")
train_words()

# ---- 3. Export to ONNX (optional — for edge deployment) ----
print("\n=== Exporting to ONNX ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def export_onnx(pth_name, json_name, onnx_name):
    pth_path  = os.path.join('models', pth_name)
    json_path = os.path.join('models', json_name)
    out_path  = os.path.join('models', onnx_name)

    if not os.path.exists(pth_path):
        print(f"  [skip] {pth_name} not found")
        return

    classes = json.load(open(json_path, encoding='utf-8'))

    # Auto-detect architecture from checkpoint
    state = torch.load(pth_path, map_location='cpu')
    first_w = next(iter(state.values()))
    input_dim   = first_w.shape[1]
    hidden_dim  = first_w.shape[0] // 4
    num_classes = state[list(state.keys())[-1]].shape[0]

    model = LSMLSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=num_classes,
    )
    model.load_state_dict(state)
    model.to(device).eval()

    dummy = torch.randn(1, 30, input_dim).to(device)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=17,
    )
    print(f"  {pth_name} -> {onnx_name}  (input={input_dim}D, classes={num_classes})")

export_onnx('dynamic_letters_model.pth', 'dynamic_letters_classes.json', 'dynamic_letters_model.onnx')
export_onnx('words_model.pth',           'words_classes.json',           'words_model.onnx')

print("\n=== Done. Copy models back to Drive: ===")
print("!cp /content/LSM/models/*.pth  /content/drive/MyDrive/LSM/models/")
print("!cp /content/LSM/models/*.json /content/drive/MyDrive/LSM/models/")
