import os
import json
import joblib
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_PATH = os.path.join('data')
STATIC_LETTERS_PATH = os.path.join(DATA_PATH, 'static_letters')
DYNAMIC_LETTERS_PATH = os.path.join(DATA_PATH, 'dynamic_letters')
WORDS_PATH = os.path.join(DATA_PATH, 'words')
MODELS_DIR = os.path.join('models')

os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------
# Bidirectional LSTM + Temporal Attention
# -------------------------------------------------------
# INPUT_DIM = 378: positions(126) + velocities(126) + accelerations(126)
# The velocity/acceleration channels give the model direct access to how fast
# each landmark moves — the primary discriminator for dynamic signs.
#
# Architecture choices:
#   - Bidirectional: forward + backward context at every frame
#   - Attention pooling: learns WHICH frames matter most for each sign
#     (e.g. the stroke peak of J vs. the direction reversal of Z)
#   - hidden_dim=256 per direction: enough capacity for 250+ classes
# -------------------------------------------------------
MOTION_INPUT_DIM = 378  # pos + vel + acc

class LSMLSTMModel(nn.Module):
    def __init__(self, input_dim=MOTION_INPUT_DIM, hidden_dim=256,
                 num_layers=2, num_classes=10, dropout=0.35):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        # Attention: learn a scalar weight per frame
        self.attn = nn.Linear(hidden_dim * 2, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)              # (B, T, hidden*2)
        w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        ctx = (w * out).sum(dim=1)         # (B, hidden*2)
        return self.head(ctx)


# -------------------------------------------------------
# Data Augmentation for sequences
# -------------------------------------------------------
def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Applies random spatial + temporal augmentation to a (T, 126) sequence.
    All augmentations preserve the meaning of the sign while increasing
    variance so the model generalises to different speeds, distances, and
    minor positional offsets.

    Augmentations applied (each independently randomised per call):
      1. Gaussian noise      — simulates landmark jitter from the ONNX detector
      2. Temporal speed jitter — resample to a random speed (0.7× – 1.3×),
                                 then crop/pad back to T frames
      3. Scale jitter         — uniform scale of hand size (0.85 – 1.15×)
      4. Wrist translation    — small random offset to wrist position
    """
    T, D = seq.shape
    out = seq.copy()

    # 1. Gaussian noise
    out += np.random.normal(0, 0.008, out.shape).astype(np.float32)

    # 2. Temporal speed jitter — resample to len in [0.7T, 1.3T], crop/pad to T
    speed = np.random.uniform(0.7, 1.3)
    new_len = max(4, int(T * speed))
    indices = np.round(np.linspace(0, T - 1, new_len)).astype(int)
    resampled = out[indices]
    if new_len >= T:
        out = resampled[:T]
    else:
        pad = np.zeros((T - new_len, D), dtype=np.float32)
        out = np.concatenate([resampled, pad], axis=0)

    # 3. Scale jitter (per-hand, affects all landmarks equally)
    scale = np.random.uniform(0.85, 1.15)
    out *= scale

    # 4. Wrist translation noise (only wrist channels 0:3 and 63:66)
    shift = np.random.normal(0, 0.02, 3).astype(np.float32)
    out[:, 0:3]   += shift
    out[:, 63:66] += shift

    return out


# -------------------------------------------------------
# Lazy Dataset — augmentation + motion features per-sample
# -------------------------------------------------------
# Memory-safe: only the base positions (N, 30, 126) stay in RAM.
# Augmentation and the velocity/acceleration channels (→ 378) are computed
# on-the-fly inside __getitem__, so a 15k-sample dataset peaks at ~2 GB
# instead of materialising the full (N*copies, 30, 378) array (~10+ GB).
def mirror_sequence(seq):
    """
    Espeja una secuencia (T,126) horizontalmente para simular la MISMA seña
    hecha con la otra mano. Esto da invarianza a lateralidad (handedness):
      - Niega la coordenada X de todos los landmarks (reflejo horizontal)
      - Intercambia mano izquierda [0:63] <-> mano derecha [63:126]
    Una seña grabada con la derecha produce así su gemela con la izquierda,
    de modo que el modelo reconoce ambas manos.
    """
    T = seq.shape[0]
    s = seq.reshape(T, 2, 21, 3).copy()   # (T, [izq,der], 21, 3)
    s[..., 0] *= -1                        # reflejo horizontal (negar X)
    s = s[:, ::-1, :, :]                   # intercambiar izquierda <-> derecha
    return np.ascontiguousarray(s.reshape(T, 126)).astype(np.float32)


def _motion_features(seq):
    """pos(126) + vel(126) + acc(126) = 378. Solo numpy (no importa cv2).
    Idéntico a utils.compute_motion_features pero sin la dependencia de cv2,
    para no romper los DataLoader workers en Windows (paging file)."""
    T = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=1).astype(np.float32)


class _LazySeqDataset(torch.utils.data.Dataset):
    def __init__(self, base_X, base_y, indices, train=False):
        self.base_X  = base_X          # (N, 30, 126) float32, shared (copy-on-write)
        self.base_y  = base_y
        self.indices = np.asarray(indices)
        self.train   = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        seq = self.base_X[idx]                     # (30, 126)
        if self.train:
            seq = augment_sequence(seq)            # random spatial/temporal jitter
        motion = _motion_features(seq)             # (30, 378) pos+vel+acc (sin cv2)
        return torch.from_numpy(motion), int(self.base_y[idx])


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
        n_jobs=1,  # joblib subprocesses import torch which OOMs the page file on Windows
        max_depth=None,
        min_samples_split=3
    )

    # Apply Platt scaling calibration with 5-fold CV
    calibrated_rf = CalibratedClassifierCV(rf, cv=5, method='sigmoid', n_jobs=1)
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
        
    import utils as _utils

    X_raw, y_raw = [], []
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(data_path, cls)
        all_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        real_files = [f for f in all_files if not f.startswith('mock_')]
        files = real_files if len(real_files) > 0 else all_files
        print(f"  {cls}: {len(real_files)} real samples (using {len(files)} total)")
        for f in files:
            filepath = os.path.join(cls_dir, f)
            try:
                sample = np.load(filepath).astype(np.float32)   # (30, 126)
                sample = _utils.normalize_sequence_wrists(sample)
                X_raw.append(sample)
                y_raw.append(class_to_idx[cls])
            except Exception:
                pass

    if len(X_raw) < 10:
        print(f"Not enough samples ({len(X_raw)}).")
        return False

    X_raw = np.array(X_raw, dtype=np.float32)   # (N, 30, 126)
    y_raw = np.array(y_raw, dtype=np.int64)

    # ---- augmentation: generate extra copies of every sample ----------------
    # For classes with few samples we augment more aggressively.
    # augment_copies=4 gives each original sample 4 noisy/speed-jittered twins.
    augment_copies = 4
    X_aug, y_aug = [X_raw], [y_raw]
    for _ in range(augment_copies):
        aug_batch = np.stack([augment_sequence(s) for s in X_raw])
        X_aug.append(aug_batch)
        y_aug.append(y_raw)
    X_all = np.concatenate(X_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)
    print(f"  Total after augmentation: {len(X_all)} samples")

    # ---- add motion features: pos + vel + acc => (N, 30, 378) --------------
    X_motion = np.stack([_utils.compute_motion_features(s) for s in X_all])
    print(f"  Feature shape: {X_motion.shape}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_motion, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # ---- WeightedRandomSampler for imbalanced classes -----------------------
    class_counts = np.bincount(y_tr)
    weights = 1.0 / class_counts[y_tr]
    sampler = WeightedRandomSampler(weights, num_samples=len(y_tr), replacement=True)

    train_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_dataset   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader  = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on: {device}")

    model = LSMLSTMModel(
        input_dim=MOTION_INPUT_DIM,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(classes),
        dropout=0.35,
    ).to(device)

    # Label smoothing reduces overconfidence and improves calibration
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Cosine annealing: smooth LR decay, avoids getting stuck in poor optima
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)

    max_epochs = 120
    patience   = 15
    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += criterion(logits, by).item()
                correct  += (logits.argmax(1) == by).sum().item()
                total    += by.size(0)

        avg_val = val_loss / len(val_loader)
        val_acc = correct / total if total else 0
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{max_epochs} | "
                  f"train={train_loss/len(train_loader):.4f} "
                  f"val={avg_val:.4f} acc={val_acc*100:.1f}% lr={lr:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    if best_state is not None:
        model.load_state_dict(best_state)

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


def train_unified():
    """
    Trains a SINGLE model covering both dynamic letters (j,k,q,x,z,ñ) and all
    LSM words. This is the correct architecture: both are identical problems
    (sequence-of-landmarks → class label) and a unified model eliminates the
    manual mode switch and the train/serve skew from having two pipelines.

    Strategy:
    - Merge data/dynamic_letters/ and data/words/ into one class space
    - Classes that appear in both (e.g. 'neutral') are merged automatically
    - Word classes (few samples) get 8x augmentation; letter classes (many
      samples) get 4x — this reduces the ~20:1 natural imbalance to ~2:1
      before WeightedRandomSampler further equalises the batches
    """
    print("\n--- Training UNIFIED Model (Dynamic Letters + Words) ---")
    import utils as _utils

    # ---- scan both data directories ----------------------------------------
    all_sources = []   # list of (class_name, file_path)
    class_sample_counts = {}

    for base_path in [DYNAMIC_LETTERS_PATH, WORDS_PATH]:
        if not os.path.isdir(base_path):
            continue
        for cls in os.listdir(base_path):
            cls_dir = os.path.join(base_path, cls)
            if not os.path.isdir(cls_dir):
                continue
            all_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
            real_files = [f for f in all_files if not f.startswith('mock_')]
            files = real_files if real_files else all_files
            if len(files) < 2:
                print(f"  [skip] {cls}: only {len(files)} samples")
                continue
            for f in files:
                all_sources.append((cls, os.path.join(cls_dir, f)))
            class_sample_counts[cls] = class_sample_counts.get(cls, 0) + len(files)

    if not all_sources:
        print("No data found.")
        return False

    # report
    print(f"  Classes: {len(class_sample_counts)}")
    print(f"  Raw samples: {len(all_sources)}")

    classes = sorted(class_sample_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # ---- load ONLY originals into RAM (no pre-augment, no pre-motion) --------
    # Memory-safe: keeps ~N×30×126 floats in RAM, NOT N×5×30×378.
    # Augmentation + motion features are computed lazily per-sample in the
    # Dataset below, so a 15k-sample dataset uses ~2 GB instead of ~46 GB.
    base_X, base_y = [], []
    for cls, fpath in all_sources:
        try:
            seq = np.load(fpath).astype(np.float32)
            seq = _utils.normalize_sequence_wrists(seq)
        except Exception:
            continue
        base_X.append(seq)
        base_y.append(class_to_idx[cls])

    base_X = np.asarray(base_X, dtype=np.float32)   # (N, 30, 126)
    base_y = np.asarray(base_y, dtype=np.int64)
    print(f"  Loaded {len(base_X)} originals into RAM ({base_X.nbytes/1e9:.2f} GB)")

    # ---- AUGMENTACIÓN DE ESPEJO: invarianza a lateralidad (ambas manos) ------
    # Duplica cada muestra con su reflejo horizontal (mano contraria), para que
    # el modelo reconozca la seña hecha con la mano izquierda O la derecha.
    mirrored = np.stack([mirror_sequence(s) for s in base_X])
    base_X = np.concatenate([base_X, mirrored], axis=0)
    base_y = np.concatenate([base_y, base_y], axis=0)
    print(f"  Con espejo (ambas manos): {len(base_X)} muestras")

    idx_tr, idx_val = train_test_split(
        np.arange(len(base_X)), test_size=0.2, random_state=42, stratify=base_y
    )

    train_ds = _LazySeqDataset(base_X, base_y, idx_tr, train=True)
    val_ds   = _LazySeqDataset(base_X, base_y, idx_val, train=False)

    # WeightedRandomSampler para balancear clases en train
    counts  = np.bincount(base_y[idx_tr], minlength=len(classes))
    weights = 1.0 / np.maximum(counts[base_y[idx_tr]], 1)
    sampler = WeightedRandomSampler(weights, num_samples=len(idx_tr), replacement=True)

    # num_workers=0 en Windows (multiprocessing con spawn da problemas);
    # 2 en Linux/Colab donde fork comparte memoria sin overhead.
    import platform
    _nw = 0 if platform.system() == 'Windows' else 2
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=_nw)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=_nw)
    print(f"  Train: {len(idx_tr)} | Val: {len(idx_val)} (augmentación lazy, workers={_nw})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = LSMLSTMModel(
        input_dim=MOTION_INPUT_DIM,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(classes),
        dropout=0.35,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)

    max_epochs, patience = 120, 15
    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(max_epochs):
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        vl, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                vl      += criterion(logits, by).item()
                correct += (logits.argmax(1) == by).sum().item()
                total   += by.size(0)

        avg_val = vl / len(val_loader)
        acc     = correct / total if total else 0
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{max_epochs} | "
                  f"train={tr_loss/len(train_loader):.4f} "
                  f"val={avg_val:.4f} acc={acc*100:.1f}%")

        if avg_val < best_val:
            best_val   = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'unified_model.pth'))
    with open(os.path.join(MODELS_DIR, 'unified_classes.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False)
    print(f"  Saved: unified_model.pth ({len(classes)} classes)")
    return True


if __name__ == '__main__':
    train_unified()
