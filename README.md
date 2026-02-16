# Multimodal Media Authenticity & Forensics System

## Overview

This project detects manipulated or deepfake media using multimodal forensic analysis including:

• Visual Frame Analysis  
• Audio Forensic Analysis  
• Metadata Integrity Analysis  
• Fusion Model Decision System  

The system combines classical forensic techniques and machine learning to produce a final authenticity score.

---

## Features

Visual Forensics
- Frame extraction
- Face anomaly detection
- Frame anomaly distribution

Audio Forensics
- Audio extraction using FFmpeg
- Spectrogram analysis
- Audio anomaly scoring

Metadata Forensics
- Resolution analysis
- FPS consistency check
- Structural anomaly detection

Fusion Engine
- Combines all forensic signals
- Machine learning decision using Random Forest

---

## Technologies Used

Python  
Streamlit  
OpenCV  
Librosa  
Scikit-learn  

---

## Dataset

Real Videos:
Real-World Dataset

Fake Videos:
Deepfake Dataset

---

## Model

Fusion Model:
Random Forest Classifier

Accuracy:
~67% (current prototype)

---

## Run Instructions

Install dependencies

