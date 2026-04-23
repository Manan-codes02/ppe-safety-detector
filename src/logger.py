# ─────────────────────────────────────────────
# logger.py — Violation Logger
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import csv
import time
from utils import get_timestamp
from config import LOG_FILE, LOG_COOLDOWN, MODEL_NAME


class ViolationLogger:
    def __init__(self):
        self.last_logged = {}  # tracks last log time per class

        # Create/open CSV file and write header
        self.csv_file = open(LOG_FILE, "a", newline="")
        self.writer   = csv.writer(self.csv_file)
        self.writer.writerow([
            "Timestamp", "Violation", "Confidence", "Model"
        ])
        print(f"[Logger] Logging violations to: {LOG_FILE}")

    def log(self, violations):
        """
        Log violations with cooldown.
        violations = list of (class_name, confidence) tuples
        """
        now = time.time()

        for cls_name, conf in violations:

            # Only log if cooldown has passed for this class
            if cls_name not in self.last_logged or \
               (now - self.last_logged[cls_name]) > LOG_COOLDOWN:

                self.writer.writerow([
                    get_timestamp(),
                    cls_name,
                    f"{conf:.2f}",
                    MODEL_NAME
                ])
                self.csv_file.flush()
                self.last_logged[cls_name] = now

                print(f"[{get_timestamp()}] "
                      f"Violation logged: {cls_name} "
                      f"(conf: {conf:.2f})")

    def close(self):
        """Close CSV file cleanly."""
        self.csv_file.close()
        print("[Logger] Log file closed.")