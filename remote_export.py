import os
import torch
import torch.onnx
import json
import sys

print("=== Starting ONNX Export ===")
os.chdir('/content/LSM')

# Import the model definition from train_models.py
sys.path.append('/content/LSM')
from train_models import LSMLSTMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def export_onnx(model_name, classes_name, output_onnx_name):
    model_path = os.path.join("models", model_name)
    classes_path = os.path.join("models", classes_name)
    output_path = os.path.join("models", output_onnx_name)
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
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
    print(f"Exported: {output_onnx_name}")

export_onnx("dynamic_letters_model.pth", "dynamic_letters_classes.json", "dynamic_letters_model.onnx")
export_onnx("words_model.pth", "words_classes.json", "words_model.onnx")

print("=== ONNX Export Complete ===")
