"""
procesar_nuevos.py — Procesa los videos de Nuevos_Dinamicos con el pipeline ONNX
de la app y genera datos limpios (30,126) + clase neutral.

Salida: data_nuevos/dynamic_letters/{j,k,q,x,z,ñ,neutral}/*.npy
"""
import os, glob, numpy as np, cv2, random
from pathlib import Path

SRC = "Nuevos_Dinamicos"
OUT = os.path.join("data_nuevos", "dynamic_letters")
os.makedirs(OUT, exist_ok=True)

LETRAS = {'J':'j', 'K':'k', 'Q':'q', 'X':'x', 'Z':'z', 'Ñ':'ñ'}
SEQ_LEN = 30
MIN_HAND_FRAMES = 8

print("Cargando detector ONNX...")
from gpu_hand_detector import GPUHandDetector
import utils
det = GPUHandDetector(palm_model_path='palm_detection.onnx',
                      hand_model_path='handpose_estimation_mediapipe_2023feb.onnx')
if not det.load():
    print("[ERROR] No cargó el detector"); raise SystemExit
print("Detector OK\n")


def procesar_video(path):
    """Video -> (30,126) con landmarks. None si <MIN_HAND_FRAMES con mano."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release(); return None
    idxs = np.linspace(0, total - 1, SEQ_LEN, dtype=int)
    seq, hands = [], 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            seq.append(np.zeros(126, dtype=np.float32)); continue
        res = det.detect(frame)
        feat = utils.extract_features(res)
        if not np.all(feat == 0):
            hands += 1
        seq.append(feat.astype(np.float32))
    cap.release()
    if hands < MIN_HAND_FRAMES:
        return None
    return np.array(seq, dtype=np.float32)


# ── 1. Procesar cada letra ────────────────────────────────────────────────
counts = {}
for carpeta, letra in LETRAS.items():
    src_dir = os.path.join(SRC, carpeta)
    if not os.path.isdir(src_dir):
        print(f"[!] No existe {src_dir}"); continue
    out_dir = os.path.join(OUT, letra)
    os.makedirs(out_dir, exist_ok=True)
    vids = glob.glob(os.path.join(src_dir, "*.mp4"))
    ok = 0
    for vp in vids:
        seq = procesar_video(vp)
        if seq is None:
            continue
        np.save(os.path.join(out_dir, f"{Path(vp).stem}.npy"), seq)
        ok += 1
    counts[letra] = ok
    print(f"  {letra}: {ok}/{len(vids)} videos procesados")

print(f"\nTotal letras procesadas: {sum(counts.values())}")

# ── 2. Generar clase NEUTRAL (entre señas / sin seña / reposo) ────────────
# Estrategia:
#   a) edge crops: primeros/últimos frames de secuencias (mano entrando/saliendo)
#   b) interpolaciones entre letras distintas (transición real)
#   c) poses estáticas con jitter (mano quieta sin señar)
print("\nGenerando clase neutral...")
random.seed(42); np.random.seed(42)

# cargar las secuencias reales procesadas
seqs = {}
for letra in counts:
    arrs = [np.load(f) for f in glob.glob(os.path.join(OUT, letra, "*.npy"))]
    if arrs:
        seqs[letra] = arrs

neutral_dir = os.path.join(OUT, "neutral")
os.makedirs(neutral_dir, exist_ok=True)
neu = []

# a) edge crops — inicio/fin con padding (mano entrando o saliendo)
for letra, arrs in seqs.items():
    for arr in arrs:
        for _ in range(2):
            n = random.randint(5, 10)
            use_start = random.random() < 0.5
            crop = arr[:n] if use_start else arr[-n:]
            pad = np.zeros((SEQ_LEN, 126), dtype=np.float32)
            if use_start: pad[:n] = crop
            else:         pad[-n:] = crop
            pad += np.random.normal(0, 0.003, pad.shape).astype(np.float32)
            neu.append(pad)

# b) interpolaciones entre 2 letras distintas
letras_list = list(seqs.keys())
for _ in range(300):
    a, b = random.sample(letras_list, 2)
    sa, sb = random.choice(seqs[a]), random.choice(seqs[b])
    t = np.linspace(0, 1, SEQ_LEN).reshape(-1, 1)
    interp = (sa * (1 - t) + sb * t).astype(np.float32)
    interp += np.random.normal(0, 0.004, interp.shape).astype(np.float32)
    neu.append(interp)

# c) poses estáticas con jitter (mano quieta)
for _ in range(300):
    letra = random.choice(letras_list)
    arr = random.choice(seqs[letra])
    base = arr[random.randint(0, SEQ_LEN - 1)]      # un frame congelado
    static = np.tile(base, (SEQ_LEN, 1)).astype(np.float32)
    static += np.random.normal(0, 0.012, static.shape).astype(np.float32)
    neu.append(static)

random.shuffle(neu)
# balancear: ~same count as average letter
target = int(np.mean(list(counts.values())))
neu = neu[:max(target, 120)]
for i, s in enumerate(neu):
    np.save(os.path.join(neutral_dir, f"neu_{i:04d}.npy"), s)

print(f"  neutral: {len(neu)} muestras generadas")
print("\n=== RESUMEN ===")
for letra in list(counts.keys()) + ["neutral"]:
    n = len(glob.glob(os.path.join(OUT, letra, "*.npy")))
    print(f"  {letra}: {n}")
print(f"\nDatos en: {os.path.abspath(OUT)}")
