import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = os.path.join('data')
STATIC_LETTERS_PATH = os.path.join(DATA_PATH, 'static_letters')
DYNAMIC_LETTERS_PATH = os.path.join(DATA_PATH, 'dynamic_letters')
WORDS_PATH = os.path.join(DATA_PATH, 'words')
MODELS_DIR = os.path.join('models')

os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------
# PyTorch LSTM Model
# -----------------
class LSMLSTMModel(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=64, num_layers=2, num_classes=5):
        super(LSMLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # out: (batch_size, seq_len, hidden_dim)
        # Take the output of the last frame in the sequence
        out = self.fc(out[:, -1, :])
        return out

# -----------------
# Synthetic Data Generator
# -----------------
def generate_synthetic_data_if_empty():
    """
    Generates synthetic landmark coordinates to enable model compilation
    and testing when no webcam captures exist.
    """
    import random
    
    # 1. Static Letters (a, b, c, d, e)
    static_classes = ['a', 'b', 'c', 'd', 'e']
    has_static_data = False
    if os.path.exists(STATIC_LETTERS_PATH):
        subdirs = [d for d in os.listdir(STATIC_LETTERS_PATH) if os.path.isdir(os.path.join(STATIC_LETTERS_PATH, d))]
        if len(subdirs) > 0:
            has_static_data = True
            
    if not has_static_data:
        print("[Synthetic] Generating mock dataset for Static Letters...")
        for cls in static_classes:
            cls_dir = os.path.join(STATIC_LETTERS_PATH, cls)
            os.makedirs(cls_dir, exist_ok=True)
            # Create a base hand posture vector of 126 coordinates
            # Most coordinates are wrist-centered normalized coords
            base_shape = np.random.normal(loc=0.0, scale=0.3, size=126)
            # Make sure it's not all zeros
            base_shape[0:3] = [0.1, -0.2, 0.0]
            
            for i in range(25): # 25 samples
                noise = np.random.normal(loc=0.0, scale=0.03, size=126)
                sample = base_shape + noise
                np.save(os.path.join(cls_dir, f"mock_{i}.npy"), sample)

    # 2. Dynamic Letters (j, k, ñ, q, x, z)
    dynamic_letters = ['j', 'k', 'ñ', 'q', 'x', 'z']
    has_dyn_letters = False
    if os.path.exists(DYNAMIC_LETTERS_PATH):
        subdirs = [d for d in os.listdir(DYNAMIC_LETTERS_PATH) if os.path.isdir(os.path.join(DYNAMIC_LETTERS_PATH, d))]
        if len(subdirs) > 0:
            has_dyn_letters = True
            
    if not has_dyn_letters:
        print("[Synthetic] Generating mock dataset for Dynamic Letters...")
        for cls in dynamic_letters:
            cls_dir = os.path.join(DYNAMIC_LETTERS_PATH, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(25): # 25 samples of 30-frame sequences
                sequence = []
                # Define a trajectory direction
                trajectory_dir = np.random.normal(loc=0.0, scale=0.1, size=3)
                base_shape = np.random.normal(loc=0.0, scale=0.3, size=126)
                base_shape[0:3] = [0.1, -0.2, 0.0]
                for f in range(30):
                    # Simulate movement of wrist (indices 0:3 and 63:66)
                    step_shape = base_shape.copy()
                    step_shape[0:3] += trajectory_dir * (f / 30.0)
                    # Add noise
                    noise = np.random.normal(loc=0.0, scale=0.02, size=126)
                    sequence.append(step_shape + noise)
                np.save(os.path.join(cls_dir, f"mock_{i}.npy"), np.array(sequence))

    # 3. Words (hola, gracias, adios, neutral)
    words = ['hola', 'gracias', 'adios', 'neutral']
    has_words = False
    if os.path.exists(WORDS_PATH):
        subdirs = [d for d in os.listdir(WORDS_PATH) if os.path.isdir(os.path.join(WORDS_PATH, d))]
        if len(subdirs) > 0:
            has_words = True
            
    if not has_words:
        print("[Synthetic] Generating mock dataset for Words...")
        for cls in words:
            cls_dir = os.path.join(WORDS_PATH, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(25):
                sequence = []
                # Complex movement trajectory
                freq = random.uniform(1.0, 3.0)
                base_shape = np.random.normal(loc=0.0, scale=0.3, size=126)
                base_shape[0:3] = [0.1, -0.2, 0.0]
                for f in range(30):
                    step_shape = base_shape.copy()
                    # Sine wave motion to represent word gestures
                    step_shape[0:3] += np.array([np.sin(f/30.0 * np.pi * freq) * 0.1, np.cos(f/30.0 * np.pi * freq) * 0.1, 0])
                    noise = np.random.normal(loc=0.0, scale=0.02, size=126)
                    sequence.append(step_shape + noise)
                np.save(os.path.join(cls_dir, f"mock_{i}.npy"), np.array(sequence))

# -----------------
# Trainer Routines
# -----------------
def train_static_letters():
    print("\n--- Training Static Letters Classifier (Random Forest + Platt Calibration) ---")
    if not os.path.exists(STATIC_LETTERS_PATH):
        print("Static letters data path does not exist.")
        return False
        
    classes = [d for d in os.listdir(STATIC_LETTERS_PATH) if os.path.isdir(os.path.join(STATIC_LETTERS_PATH, d))]
    if not classes:
        print("No classes found.")
        return False
        
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(STATIC_LETTERS_PATH, cls)
        # Only use real samples (exclude mock/synthetic files)
        all_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy') and not f.startswith('mock_')]
        files = all_files
        
        if len(files) < 2:
            print(f"Class '{cls}' has fewer than 2 real samples ({len(files)} total). Skipping.")
            continue
            
        # Subsample to a maximum of 2000 samples per class for balance
        if len(files) > 2000:
            import random
            random.seed(42)
            files = random.sample(files, 2000)
            
        print(f"Class '{cls}': {len(files)} samples.")
        for f in files:
            filepath = os.path.join(cls_dir, f)
            try:
                sample = np.load(filepath)
                if sample.shape == (126,):
                    X.append(sample)
                    y.append(cls)
                else:
                    pass  # Skip malformed samples
            except Exception:
                pass  # Skip corrupted files
            
    if len(X) < 10:
        print(f"Not enough samples ({len(X)} found, minimum required is 10).")
        return False
        
    X, y = np.array(X), np.array(y)
    print(f"\nTotal training data: {len(X)} samples across {len(set(y))} classes")
    
    # Zero out wrists to match live app and ensure absolute wrist position invariance
    X[:, 0:3] = 0.0
    X[:, 63:66] = 0.0

    # Enrich features with geometric features (39 per hand = 78 engineered + 126 raw = 204 total)
    import utils
    X = utils.engineer_static_features(X)
    print(f"Feature vector size: {X.shape[1]} dimensions")
    
    # Split: 60% train, 20% calibration, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    
    print(f"Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")
    
    # Train Random Forest with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=3
    )
    
    # Apply Platt scaling calibration with 5-fold CV
    calibrated_rf = CalibratedClassifierCV(rf, cv=5, method='sigmoid', n_jobs=-1)
    calibrated_rf.fit(X_trainval, y_trainval)
    
    # Evaluate on test set
    y_pred = calibrated_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nCalibrated Random Forest Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Save the calibrated model
    model_path = os.path.join(MODELS_DIR, 'static_letters_classifier.joblib')
    joblib.dump(calibrated_rf, model_path)
    print(f"Calibrated model saved to {model_path}")
    
    # Save class list
    classes_path = os.path.join(MODELS_DIR, 'static_letters_classes.json')
    with open(classes_path, 'w') as json_file:
        json.dump(calibrated_rf.classes_.tolist(), json_file)
    print(f"Class list saved to {classes_path}")
    return True
 
def _train_lstm_generic(data_path, model_name, json_name):
    if not os.path.exists(data_path):
        print(f"Path {data_path} does not exist.")
        return False
        
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not classes:
        print("No classes found.")
        return False
        
    # Filter classes with at least 2 samples
    valid_classes = []
    for cls in classes:
        cls_dir = os.path.join(data_path, cls)
        all_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        real_files = [f for f in all_files if not f.startswith('mock_')]
        files = real_files if len(real_files) > 0 else all_files
        if len(files) >= 2:
            valid_classes.append(cls)
        else:
            print(f"Class '{cls}' has fewer than 2 samples ({len(files)} total). Skipping from training.")
            
    classes = valid_classes
    if not classes:
        print("No valid classes with >= 2 samples found.")
        return False
        
    X, y = [], []
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        cls_dir = os.path.join(data_path, cls)
        all_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        real_files = [f for f in all_files if not f.startswith('mock_')]
        files = real_files if len(real_files) > 0 else all_files
        print(f"Class '{cls}': Found {len(real_files)} real samples (using {len(files)} total).")
        for f in files:
            filepath = os.path.join(cls_dir, f)
            sample = np.load(filepath) # (30, 126)
            import utils
            sample = utils.normalize_sequence_wrists(sample)
            X.append(sample)
            y.append(class_to_idx[cls])
            
    if len(X) < 10:
        print(f"Not enough samples ({len(X)} found, minimum required is 10).")
        return False
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = LSMLSTMModel(input_dim=126, hidden_dim=64, num_layers=2, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 30
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = correct / total if total > 0 else 0
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - Val Acc: {val_acc*100:.2f}%")
            
    # Save model and json classes
    model_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    mapping_path = os.path.join(MODELS_DIR, json_name)
    inv_class_to_idx = {v: k for k, v in class_to_idx.items()}
    class_list = [inv_class_to_idx[i] for i in range(len(classes))]
    with open(mapping_path, 'w') as json_file:
        json.dump(class_list, json_file)
    print(f"Class list saved to {mapping_path}")
    return True

def train_dynamic_letters():
    print("\n--- Training Dynamic Letters Classifier (LSTM in PyTorch) ---")
    return _train_lstm_generic(DYNAMIC_LETTERS_PATH, 'dynamic_letters_model.pth', 'dynamic_letters_classes.json')

def train_words():
    print("\n--- Training Words Classifier (LSTM in PyTorch) ---")
    return _train_lstm_generic(WORDS_PATH, 'words_model.pth', 'words_classes.json')

if __name__ == '__main__':
    # No longer generating synthetic data — using real + archive data only
    train_static_letters()
    train_dynamic_letters()
    train_words()
