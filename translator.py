"""
translator.py — LSM unified translator
=======================================
Un solo modelo LSTM cubre letras dinámicas (j,k,q,x,z,ñ) y palabras LSM.
No hay modo manual — el modelo detecta automáticamente qué se está señando.
"""

import os
import json
import numpy as np
import torch
from collections import deque
import utils

from train_models import LSMLSTMModel, MOTION_INPUT_DIM


class LSMTranslator:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        # LSTM en GPU si está disponible — el modelo es pequeño (11MB),
        # entra bien en 2GB compartiendo con el detector ONNX en DML.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model   = None
        self.classes = []

        # Ventana deslizante: últimos 30 frames × 126-D
        self.sequence_len = 30
        self.sequence = deque(maxlen=self.sequence_len)

        # Debouncing: confirmar predicción estable durante 8 frames
        self.pred_history   = deque(maxlen=10)
        self.last_committed = None
        self.no_hand_counter = 0

        # Carry-over de oclusión por mano
        self.last_valid_left  = None
        self.last_valid_right = None
        self.left_lost_frames  = 0
        self.right_lost_frames = 0

        # Cooldown tras commit de palabra (evita doble disparo)
        self.frame_counter   = 0
        self.last_lstm_pred  = (None, None, 0.0)
        self.commit_cooldown = 0

        self.load_model()

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def load_model(self):
        pth  = os.path.join(self.models_dir, 'unified_model.pth')
        jsn  = os.path.join(self.models_dir, 'unified_classes.json')

        # Fallback: si no existe el unificado, intenta cargar el de letras dinámicas
        if not os.path.exists(pth):
            pth = os.path.join(self.models_dir, 'dynamic_letters_model.pth')
            jsn = os.path.join(self.models_dir, 'dynamic_letters_classes.json')

        if not (os.path.exists(pth) and os.path.exists(jsn)):
            print("[Translator] No se encontró ningún modelo entrenado.")
            return

        try:
            with open(jsn, 'r', encoding='utf-8') as f:
                self.classes = json.load(f)

            state     = torch.load(pth, map_location='cpu')
            first_w   = next(iter(state.values()))
            input_dim = first_w.shape[1]
            hidden    = first_w.shape[0] // 4
            n_classes = state[list(state.keys())[-1]].shape[0]

            self.model = LSMLSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden,
                num_layers=2,
                num_classes=n_classes,
            )
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
            print(f"[Translator] Modelo cargado: {n_classes} clases, "
                  f"input={input_dim}D  ({os.path.basename(pth)})")
        except Exception as e:
            print(f"[Translator] Error cargando modelo: {e}")

    # ------------------------------------------------------------------
    # Inferencia LSTM
    # ------------------------------------------------------------------

    def _predict(self, threshold=0.85, margin=0.30):
        """
        Predice la seña actual con triple filtro anti-falsos-positivos:
          1. Clase 'neutral' entrenada (transiciones/reposo) -> nada
          2. Confianza mínima (threshold): si la mejor opción no supera el
             umbral, no es nada. Sube a 0.85 para letras (son distintivas).
          3. Margen: si la 1ª y 2ª clase más probables están demasiado cerca,
             la predicción es ambigua (típico ENTRE señas) -> nada.
        Devuelve (committed, raw, conf). committed=None significa "no es nada".
        """
        if self.model is None or len(self.sequence) < 15:
            return None, None, 0.0

        seq = np.array(self.sequence, dtype=np.float32)
        seq = utils.normalize_sequence_wrists(seq)
        if len(self.sequence) < self.sequence_len:
            pad = np.zeros((self.sequence_len - len(self.sequence), 126), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)

        motion = utils.compute_motion_features(seq)

        with torch.no_grad():
            t     = torch.tensor(motion).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(t), dim=1)[0]

        # top-2 para el chequeo de margen
        top = torch.topk(probs, k=min(2, probs.shape[0]))
        conf  = top.values[0].item()
        idx   = top.indices[0].item()
        second = top.values[1].item() if probs.shape[0] > 1 else 0.0
        label = self.classes[idx]

        # 1. neutral entrenada
        if label == 'neutral':
            return None, None, 0.0
        # 2. confianza mínima
        if conf < threshold:
            return None, label, conf
        # 3. margen: 1ª vs 2ª demasiado cerca = ambiguo = entre señas
        if (conf - second) < margin:
            return None, label, conf
        return label, label, conf

    # ------------------------------------------------------------------
    # Entrada principal por frame
    # ------------------------------------------------------------------

    def process_frame(self, features, mode=None):
        """
        Args:
            features : np.ndarray (126,) — landmarks de la pipeline de cámara
            mode     : ignorado — modelo unificado detecta todo automáticamente
        Returns:
            (committed, raw, confidence)
            committed != None cuando hay una predicción estable lista para emitir.
        """
        # A. Carry-over por oclusión (hasta 15 frames por mano)
        if not np.all(features == 0):
            left  = features[0:63]
            right = features[63:126]

            if np.all(left == 0):
                self.left_lost_frames += 1
                if self.left_lost_frames <= 15 and self.last_valid_left is not None:
                    left = self.last_valid_left
            else:
                self.left_lost_frames = 0
                self.last_valid_left  = left.copy()

            if np.all(right == 0):
                self.right_lost_frames += 1
                if self.right_lost_frames <= 15 and self.last_valid_right is not None:
                    right = self.last_valid_right
            else:
                self.right_lost_frames = 0
                self.last_valid_right  = right.copy()

            features = np.concatenate([left, right])

        # B. Reset si la mano desaparece más de 1 segundo
        if np.all(features == 0):
            self.no_hand_counter += 1
            if self.no_hand_counter > 30:
                self._reset()
            elif self.sequence:
                self.sequence.append(self.sequence[-1])
            return None, None, 0.0

        self.no_hand_counter = 0
        self.sequence.append(features.copy())
        self.frame_counter += 1

        # C. LSTM cada 3 frames para cubrir gestos rápidos
        if self.frame_counter % 3 == 0:
            self.last_lstm_pred = self._predict()

        pred, raw, conf = self.last_lstm_pred

        if pred is None:
            # Entre señas / sin certeza: limpiar historial para no arrastrar,
            # y permitir que la MISMA letra se repita después de un hueco.
            self.pred_history.clear()
            self.last_committed = None
            return None, raw, conf

        # D. Debouncing: 8 frames consecutivos idénticos → commit
        self.pred_history.append(pred)
        recent = list(self.pred_history)[-8:]
        if len(recent) == 8 and len(set(recent)) == 1:
            if pred != self.last_committed:
                self.last_committed = pred
                return pred, raw, conf

        return None, raw, conf

    def _reset(self):
        self.sequence.clear()
        self.last_committed   = None
        self.last_valid_left  = None
        self.last_valid_right = None
        self.last_lstm_pred   = (None, None, 0.0)
        self.commit_cooldown  = 0
