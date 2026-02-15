import joblib
import numpy as np

# Load trained model once
try:
    model = joblib.load("fusion_model.pkl")
except:
    model = None


def fuse_results(visual_mean, visual_variance, suspicious_ratio, audio_score, metadata_score):

    if model is None:
        return {
            "final_score": 0.0,
            "decision": "Model not loaded"
        }

    features = np.array([[ 
        visual_mean,
        visual_variance,
        suspicious_ratio,
        audio_score,
        metadata_score
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # probability of FAKE

    decision = "FAKE" if prediction == 1 else "REAL"

    return {
        "final_score": float(probability),
        "decision": decision
    }
