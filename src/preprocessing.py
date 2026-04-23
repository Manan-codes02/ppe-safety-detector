# ─────────────────────────────────────────────
# preprocessing.py — Input Data Preparation
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import cv2
from config import IMAGE_SIZE


# ── Resize Frame ──────────────────────────────
def resize_frame(frame, size=IMAGE_SIZE):
    """
    Resize frame to target size for model input.
    Maintains aspect ratio with padding.
    """
    return cv2.resize(frame, (size, size))


# ── Normalize Frame ───────────────────────────
def normalize_frame(frame):
    """
    Normalize pixel values from 0-255 to 0-1.
    Used internally by YOLOv8 automatically,
    kept here for reference and custom pipelines.
    """
    return frame / 255.0


# ── Flip Frame ────────────────────────────────
def flip_frame(frame, flip_code=1):
    """
    Flip frame horizontally (mirror effect).
    flip_code:
        1  = horizontal flip (mirror)
        0  = vertical flip
       -1  = both
    """
    return cv2.flip(frame, flip_code)


# ── Convert Color ─────────────────────────────
def convert_bgr_to_rgb(frame):
    """
    Convert BGR (OpenCV default) to RGB.
    YOLOv8 handles this internally,
    kept here for custom use cases.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Enhance Brightness ────────────────────────
def enhance_brightness(frame, alpha=1.2, beta=10):
    """
    Improve brightness and contrast for
    low-light construction site conditions.
    alpha = contrast (1.0 = no change)
    beta  = brightness (0 = no change)
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# ── Main Preprocess Function ──────────────────
def preprocess_frame(frame, enhance=False, mirror=False):
    """
    Full preprocessing pipeline for one frame.

    Steps:
    1. Optional mirror flip
    2. Optional brightness enhancement
    3. Return processed frame

    Note: YOLOv8 handles resize and normalize
    internally before inference.
    """
    if mirror:
        frame = flip_frame(frame)

    if enhance:
        frame = enhance_brightness(frame)

    return frame


# ── Open Input Source ─────────────────────────
def open_source(source):
    """
    Open video capture from webcam or video file.
    source = 0 (webcam) or "path/to/video.mp4"
    Returns cap object.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Cannot open source: {source}")

    # Get source properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"[Source] Opened: {source}")
    print(f"[Source] Resolution: {width}x{height} @ {fps:.1f}FPS")

    return cap