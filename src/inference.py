# ─────────────────────────────────────────────
# inference.py — Model Inference Pipeline
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import time
from ultralytics import YOLO
from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD, IMAGE_SIZE
)
from utils import check_violations


class PPEDetector:
    def __init__(self):
        """Load trained YOLOv8 model."""
        print(f"[Inference] Loading model from: {MODEL_PATH}")
        self.model          = YOLO(MODEL_PATH)
        self.inference_times = []
        print("[Inference] Model loaded successfully!")

    def run(self, frame):
        """
        Run inference on a single frame.

        Steps:
        1. Record start time
        2. Run model prediction
        3. Record end time
        4. Extract violations
        5. Return results

        Returns:
            annotated_frame : frame with boxes drawn
            violations      : list of (class, confidence)
            inference_ms    : time taken in milliseconds
        """
        # ── Time the inference ────────────────
        start = time.time()

        results = self.model.predict(
            frame,
            conf    = CONFIDENCE_THRESHOLD,
            iou     = IOU_THRESHOLD,
            imgsz   = IMAGE_SIZE,
            verbose = False
        )

        inference_ms = (time.time() - start) * 1000
        self.inference_times.append(inference_ms)

        # ── Draw boxes on frame ───────────────
        annotated_frame = results[0].plot()

        # ── Extract violations ─────────────────
        violations = check_violations(results)

        return annotated_frame, violations, inference_ms

    def get_avg_inference(self):
        """Return average inference time in ms."""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)

    def get_avg_fps(self):
        """Return average FPS based on inference times."""
        avg = self.get_avg_inference()
        if avg == 0:
            return 0
        return 1000 / avg

    def print_stats(self):
        """Print final inference statistics."""
        print("\n" + "=" * 50)
        print("  Inference Statistics")
        print("=" * 50)
        print(f"  Total frames   : {len(self.inference_times)}")
        print(f"  Avg Inference  : {self.get_avg_inference():.1f} ms")
        print(f"  Avg FPS        : {self.get_avg_fps():.1f}")
        print(f"  Min Inference  : {min(self.inference_times):.1f} ms")
        print(f"  Max Inference  : {max(self.inference_times):.1f} ms")
        print("=" * 50)