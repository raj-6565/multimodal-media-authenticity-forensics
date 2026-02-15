import cv2
import os
import tempfile

def extract_frames(video_path, frame_interval=10):
    """
    Extract frames from video at a fixed interval.
    Returns list of frame file paths.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(temp_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()

    return frame_paths
import cv2
import os
import tempfile

def extract_frames(video_path, frame_interval=10):
    """
    Extract frames from video at a fixed interval.
    Returns list of frame file paths.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(temp_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()

    return frame_paths
