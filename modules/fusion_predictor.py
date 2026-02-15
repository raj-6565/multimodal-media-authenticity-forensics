import joblib
import numpy as np
import os

MODEL_PATH = "fusion_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def predict_fusion(visual_score, audio_score, metadata_score):
    model = load_model()
    
    if model is None:
        return None, "Fusion model not trained yet."

    features = np.array([[visual_score, audio_score, metadata_score]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    return prediction, float(probability)
