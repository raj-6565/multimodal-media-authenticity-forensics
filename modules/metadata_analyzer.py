import cv2

def analyze_metadata(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()

    score = 0
    score += min(fps / 120, 1) * 0.3
    score += min((width * height) / (4000 * 4000), 1) * 0.3
    score += min(frame_count / 10000, 1) * 0.4

    return {
        "fps": fps,
        "frame_count": frame_count,
        "resolution": f"{int(width)}x{int(height)}",
        "anomaly_score": score
    }
