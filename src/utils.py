# ─────────────────────────────────────────────
# utils.py — Helper Functions
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import cv2
import time
import winsound
from datetime import datetime
from config import (
    SHOW_FPS, SHOW_MODEL_NAME, SHOW_TIMESTAMP,
    MODEL_NAME, BEEP_FREQUENCY, BEEP_DURATION,
    ALERT_COOLDOWN, VIOLATION_CLASSES
)


# ── FPS Calculator ────────────────────────────
class FPSCounter:
    def __init__(self):
        self.start_time  = time.time()
        self.frame_count = 0
        self.fps         = 0

    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps         = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time  = time.time()
        return self.fps


# ── Draw UI Overlay on Frame ──────────────────
def draw_overlay(frame, fps, violation_found):
    """Draw status, FPS, model name and timestamp on frame."""

    # Violation status
    if violation_found:
        cv2.putText(frame, "WARNING: VIOLATION DETECTED!",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "ALL CLEAR",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    # FPS counter
    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

    # Model name
    if SHOW_MODEL_NAME:
        cv2.putText(frame, f"Model: {MODEL_NAME}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

    # Timestamp
    if SHOW_TIMESTAMP:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts,
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

    return frame


# ── Beep Alert ────────────────────────────────
class AlertManager:
    def __init__(self):
        self.last_alert_time = 0

    def trigger(self):
        """Beep only if cooldown has passed."""
        now = time.time()
        if now - self.last_alert_time > ALERT_COOLDOWN:
            winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)
            self.last_alert_time = now


# ── Check Violations in Results ───────────────
def check_violations(results):
    """
    Returns list of (class_name, confidence) tuples
    for any detected violation classes.
    """
    violations = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            conf     = float(box.conf)
            if cls_name in VIOLATION_CLASSES:
                violations.append((cls_name, conf))
    return violations


# ── Format Timestamp ──────────────────────────
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Print System Info ─────────────────────────
def print_system_info():
    print("=" * 50)
    print("  PPE Safety Compliance Detection System")
    print("=" * 50)
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Conf      : {0.35}")
    print(f"  Violations: {VIOLATION_CLASSES}")
    print("=" * 50)