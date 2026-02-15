import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import joblib

from modules.frame_extractor import extract_frames
from modules.frame_analyzer import analyze_frame
from modules.audio_analyzer import analyze_audio
from modules.metadata_analyzer import analyze_metadata
from modules.fusion_engine import fuse_results

st.set_page_config(layout="wide", page_title="Media Authenticity Forensics")

st.title("🛡 Multimodal Media Authenticity & Forensic Dashboard")

st.sidebar.header("Upload & Controls")

uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
run_analysis = st.sidebar.button("Run Full Analysis")

if uploaded_video and run_analysis:

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_video.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    tabs = st.tabs([
        "🎥 Visual Analysis",
        "🎧 Audio Analysis",
        "📄 Metadata Analysis",
        "🧠 Final Report"
    ])

    visual_summary = None
    audio_summary = None
    metadata_summary = None

    # ================= VISUAL TAB =================
    with tabs[0]:

        st.subheader("Visual Frame Integrity Analysis")

        frames = extract_frames(video_path)
        scores = []

        for frame in frames:
            score = analyze_frame(frame)
            if score is not None:
                scores.append(score)

        if scores:

            mean_score = float(np.mean(scores))
            variance_score = float(np.var(scores))
            suspicious_ratio = len([s for s in scores if s > 0.6]) / len(scores)

            visual_summary = {
                "mean": mean_score,
                "variance": variance_score,
                "ratio": suspicious_ratio
            }

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Visual Anomaly Score", round(mean_score, 4))
            col2.metric("Variance of Frame Scores", round(variance_score, 4))
            col3.metric("Suspicious Frame Ratio", round(suspicious_ratio, 2))

            st.markdown("### Frame Anomaly Score Distribution")

            fig, ax = plt.subplots()
            ax.hist(scores, bins=20, color="#2E86C1", edgecolor="black")

            ax.set_xlabel("Frame Anomaly Score (Higher = More Suspicious)")
            ax.set_ylabel("Number of Frames")
            ax.set_title("Distribution of Frame-level Integrity Scores")

            st.pyplot(fig)

            sorted_frames = sorted(zip(frames, scores), key=lambda x: x[1], reverse=True)

            st.markdown("### Top Suspicious Frames Identified")

            for frame_path, score in sorted_frames[:3]:
                st.image(frame_path, caption=f"Anomaly Score: {round(score,4)}", width=300)

        else:
            st.warning("No valid frames could be analyzed.")

    # ================= AUDIO TAB =================
    with tabs[1]:

        st.subheader("Audio Authenticity Analysis")

        audio_result = analyze_audio(video_path)

        if audio_result["audio_available"]:

            audio_summary = audio_result["confidence"]

            st.metric("Audio Anomaly Score", round(audio_summary, 4))

            st.markdown(
                "Spectral analysis based on MFCC, spectral centroid, and temporal frequency "
                "patterns evaluates acoustic authenticity."
            )

            if "spectrogram" in audio_result:
                st.image(audio_result["spectrogram"], caption="Audio Spectrogram")

        else:

            audio_summary = 0.0

            st.warning(
                "No audio track detected in this video. Audio modality excluded from fusion analysis."
            )

    # ================= METADATA TAB =================
    with tabs[2]:

        st.subheader("Metadata Structural Integrity Analysis")

        metadata_result = analyze_metadata(video_path)

        metadata_summary = metadata_result["anomaly_score"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Frames Per Second", metadata_result["fps"])
        col2.metric("Resolution", metadata_result["resolution"])
        col3.metric("Total Frame Count", metadata_result["frame_count"])

        st.metric("Metadata Integrity Score", round(metadata_summary, 4))

        st.markdown(
            "Metadata analysis evaluates structural consistency across encoding parameters "
            "and detects anomalies introduced during manipulation or re-encoding."
        )

    # ================= FINAL REPORT TAB =================
    with tabs[3]:

        st.subheader("Multimodal AI Fusion Assessment")

        if visual_summary and metadata_summary is not None:

            visual_mean = visual_summary["mean"]
            visual_var = visual_summary["variance"]
            visual_ratio = visual_summary["ratio"]

            model_path = "fusion_model.pkl"

            if os.path.exists(model_path):

                model = joblib.load(model_path)

                features = np.array([[
                    visual_mean,
                    visual_var,
                    visual_ratio,
                    audio_summary,
                    metadata_summary
                ]])

                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                confidence_score = probabilities[prediction]

                decision = (
                    "Manipulated / Synthetic"
                    if prediction == 1 else
                    "Authentic / Original"
                )

                st.progress(min(confidence_score, 1.0))

                col1, col2 = st.columns(2)

                col1.metric("AI Confidence Score", round(confidence_score, 4))
                col2.metric("Final System Decision", decision)

                st.markdown("### Detailed Multimodal Feature Analysis")

                st.markdown(f"""
                **Visual Forensics**
                - Mean anomaly score: **{visual_mean:.4f}**
                - Variance in frame integrity: **{visual_var:.4f}**
                - Suspicious frame proportion: **{visual_ratio:.2f}**

                **Audio Forensics**
                - Acoustic anomaly score: **{audio_summary:.4f}**

                **Metadata Forensics**
                - Structural anomaly score: **{metadata_summary:.4f}**
                """)

                # FEATURE IMPORTANCE RESTORED
                if hasattr(model, "feature_importances_"):

                    st.markdown("### Feature Contribution to Final AI Decision")

                    feature_names = [
                        "Visual Mean",
                        "Visual Variance",
                        "Suspicious Ratio",
                        "Audio Score",
                        "Metadata Score"
                    ]

                    importances = model.feature_importances_

                    fig, ax = plt.subplots()

                    ax.barh(feature_names, importances, color="#1ABC9C")

                    ax.set_xlabel("Relative Importance")
                    ax.set_title("Feature Importance in Fusion Model")

                    st.pyplot(fig)

            else:

                st.warning("Fusion model not found.")

        else:

            st.warning("Insufficient data for fusion analysis.")
