import os 

import cv2

def query(path):
    cap = cv2.VideoCapture(path)
    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    except AttributeError: # version difference
        frame_count = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)

    _, frame_1 = cap.read()
    cap.release()

    return cv2.imencode('.jpg', frame_1)[1].tobytes()
    # return f"This runner gets path: {path},  size {os.path.getsize(path) / 1e6} MB, {frame_count} frames, {width}x{height}"
