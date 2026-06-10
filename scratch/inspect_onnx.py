import onnxruntime as ort

def inspect_model(path):
    print(f"\n=== Inspecting: {path} ===")
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print("Inputs:")
        for i in session.get_inputs():
            print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        print("Outputs:")
        for o in session.get_outputs():
            print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
    except Exception as e:
        print(f"Error inspecting {path}: {e}")

inspect_model("palm_detection.onnx")
inspect_model("handpose_estimation_mediapipe_2023feb.onnx")
