import os
import sys
import numpy as np

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translator import LSMTranslator

def test_pipeline():
    print("=== Verification of LSM Translation Pipeline ===")
    
    # 1. Initialize translator
    translator = LSMTranslator(models_dir='models')
    
    # 2. Check model loading status
    has_static = translator.static_classifier is not None
    has_dyn_let = translator.dynamic_letters_model is not None
    has_words = translator.words_model is not None
    
    print(f"Loaded Static Letters Classifier: {has_static}")
    print(f"Loaded Dynamic Letters Model: {has_dyn_let}")
    print(f"Loaded Words Model: {has_words}")
    
    if not (has_static and has_dyn_let and has_words):
        print("[FAIL] Not all models were successfully loaded!")
        sys.exit(1)
        
    print("[OK] All three models loaded successfully!")
    
    # 3. Simulate processing frame coordinates
    # Generate random features representing active hand landmarks
    dummy_features = np.random.normal(loc=0.0, scale=0.1, size=126)
    
    # Run in static (Letters) mode
    print("Running static letters mode inference...")
    for i in range(40):
        # Add random wrist coordinates to simulate movement or stillness
        # Static letters require stable landmarks over time
        new_token, raw_pred, conf = translator.process_frame(dummy_features, mode='static')
        
    # Run in dynamic (Words) mode
    print("Running dynamic words mode inference...")
    for i in range(40):
        new_token, raw_pred, conf = translator.process_frame(dummy_features, mode='dynamic')
        
    print("[OK] Simulation of both modes ran successfully without any exceptions!")

if __name__ == '__main__':
    test_pipeline()
