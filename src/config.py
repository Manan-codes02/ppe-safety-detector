# ─────────────────────────────────────────────
# config.py — Central Configuration File
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import os

# ── Base Project Path ─────────────────────────
BASE_DIR = r"C:\Users\Manan Singla\Desktop\ppe_project"

# ── Model Settings ────────────────────────────
MODEL_NAME    = "yolov8n"          # "yolov8n" or "yolov8s"
MODEL_PATH    = os.path.join(BASE_DIR, "models", MODEL_NAME, "best.pt")

# ── Input Source ──────────────────────────────
# 0          = laptop webcam
# 1          = external camera
# "path/to/video.mp4" = video file
#INPUT_SOURCE  = r"C:\Users\Manan Singla\Desktop\ppe_project\videos\test_video.f234.mp4"
INPUT_SOURCE = 0

# ── Detection Settings ────────────────────────
CONFIDENCE_THRESHOLD = 0.35        # minimum confidence to show detection
IOU_THRESHOLD        = 0.45        # overlap threshold for NMS
IMAGE_SIZE           = 640         # input image size for model

# ── Violation Classes ─────────────────────────
VIOLATION_CLASSES = [
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask"
]

# ── Alert Settings ────────────────────────────
ALERT_COOLDOWN   = 3               # seconds between beep alerts
BEEP_FREQUENCY   = 1000            # Hz
BEEP_DURATION    = 500             # milliseconds

# ── Logging Settings ──────────────────────────
LOG_FILE         = os.path.join(BASE_DIR, "logs", "violations_log.csv")
LOG_COOLDOWN     = 10              # seconds between logging same violation

# ── Display Settings ──────────────────────────
WINDOW_TITLE     = "PPE Safety Detector — Press Q or ESC to quit"
SHOW_FPS         = True
SHOW_MODEL_NAME  = True
SHOW_TIMESTAMP   = True

# ── Output / Results ──────────────────────────
RESULTS_DIR      = os.path.join(BASE_DIR, "results")