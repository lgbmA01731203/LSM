"""
train_solo_letras.py — Entrena el modelo SOLO con letras dinámicas.
7 clases: j, k, q, x, z, ñ + neutral. Sin palabras, sin MSL-150.
Corre: python train_solo_letras.py
"""
import os, shutil
import train_models


def main():
    # Entrenar SOLO con los videos nuevos procesados (data_nuevos), sin palabras
    train_models.DYNAMIC_LETTERS_PATH = os.path.join('data_nuevos', 'dynamic_letters')
    train_models.WORDS_PATH = os.path.join('data', '_vacio')
    os.makedirs(train_models.WORDS_PATH, exist_ok=True)

    print("=== Entrenando SOLO letras dinámicas NUEVAS (7 clases) ===")
    train_models.train_unified()

    shutil.rmtree(train_models.WORDS_PATH, ignore_errors=True)
    print("\nListo. Modelo en models/unified_model.pth")


if __name__ == '__main__':
    main()
