import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import tempfile

def analyze_audio(video_path):

    try:

        y, sr = librosa.load(video_path, sr=None)

        # Detect missing or silent audio
        if y is None or len(y) == 0 or np.max(np.abs(y)) < 0.001:

            return {
                "audio_available": False,
                "confidence": None,
                "message": "No audio track detected in this video"
            }

        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 0.5:

            return {
                "audio_available": False,
                "confidence": None,
                "message": "Audio too short for reliable analysis"
            }

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        mfcc_std = float(np.std(mfcc))
        centroid_mean = float(np.mean(spectral_centroid))
        zcr_mean = float(np.mean(zcr))

        mfcc_score = min(mfcc_std / 50, 1.0)
        centroid_score = min(centroid_mean / 3000, 1.0)
        zcr_score = min(zcr_mean / 0.1, 1.0)

        confidence = (
            0.4 * mfcc_score +
            0.3 * centroid_score +
            0.3 * zcr_score
        )

        confidence = float(min(max(confidence, 0.01), 0.99))

        fig, ax = plt.subplots()

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(
            S_dB,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )

        ax.set_title("Audio Spectrogram")

        spec_path = os.path.join(
            tempfile.gettempdir(),
            "spectrogram.png"
        )

        plt.savefig(spec_path)
        plt.close(fig)

        return {
            "audio_available": True,
            "confidence": confidence,
            "spectrogram": spec_path,
            "duration": duration
        }

    except Exception as e:

        return {
            "audio_available": False,
            "confidence": None,
            "message": str(e)
        }
