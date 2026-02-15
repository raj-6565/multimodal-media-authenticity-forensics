import csv
import os

CSV_FILE = "fusion_training_data.csv"

def log_features(features, label):
    """
    Save extracted features + label into CSV.
    Label: 0 = Real, 1 = Fake
    """

    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "visual_mean",
                "visual_variance",
                "suspicious_ratio",
                "audio_score",
                "metadata_score",
                "label"
            ])

        writer.writerow([
            features["visual_mean"],
            features["visual_variance"],
            features["suspicious_ratio"],
            features["audio_score"],
            features["metadata_score"],
            label
        ])
