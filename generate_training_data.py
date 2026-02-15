import os
import numpy as np
import pandas as pd

from modules.frame_extractor import extract_frames
from modules.frame_analyzer import analyze_frame
from modules.audio_analyzer import analyze_audio
from modules.metadata_analyzer import analyze_metadata

DATASET_PATH = "dataset"
OUTPUT_FILE = "fusion_training_data.csv"

def extract_features(video_path):

    # VISUAL
    frames = extract_frames(video_path)
    scores = []

    for frame in frames:
        score = analyze_frame(frame)
        scores.append(score)

    if len(scores) == 0:
        return None

    visual_mean = float(np.mean(scores))
    visual_variance = float(np.var(scores))
    suspicious_ratio = len([s for s in scores if s > 0.6]) / len(scores)

    # AUDIO
    audio_result = analyze_audio(video_path)
    audio_score = audio_result["confidence"]

    # METADATA
    metadata_result = analyze_metadata(video_path)
    metadata_score = metadata_result["anomaly_score"]

    return {
        "visual_mean": visual_mean,
        "visual_variance": visual_variance,
        "suspicious_ratio": suspicious_ratio,
        "audio_score": audio_score,
        "metadata_score": metadata_score
    }


data_rows = []

for label_folder in ["real", "fake"]:

    folder_path = os.path.join(DATASET_PATH, label_folder)
    label = 0 if label_folder == "real" else 1

    for file in os.listdir(folder_path):
        if file.endswith((".mp4", ".avi", ".mov")):

            video_path = os.path.join(folder_path, file)
            print("Processing:", video_path)

            features = extract_features(video_path)

            if features is not None:
                features["label"] = label
                data_rows.append(features)


df = pd.DataFrame(data_rows)
df.to_csv(OUTPUT_FILE, index=False)

print("Training dataset generated successfully.")
