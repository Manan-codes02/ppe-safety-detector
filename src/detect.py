from ultralytics import YOLO
import cv2
import csv
import winsound
import time
from datetime import datetime

# ── Model Selection ──────────────────────────────
# Change "yolov8n" to "yolov8s" to switch models
MODEL = "yolov8n"
model = YOLO(rf"C:\Users\Manan Singla\Desktop\ppe_project\models\{MODEL}\best.pt")

# ── Settings ─────────────────────────────────────
VIOLATION_CLASSES = ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]
CONFIDENCE        = 0.35
LOG_COOLDOWN      = 10   # seconds between logging same violation
ALERT_COOLDOWN    = 3    # seconds between beeps

# ── CSV Logger ────────────────────────────────────
csv_file = open(
    r"C:\Users\Manan Singla\Desktop\ppe_project\logs\violations_log.csv",
    "a", newline=""
)
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Violation", "Confidence", "Model"])

# ── Webcam ────────────────────────────────────────
cap = cv2.VideoCapture(0)
print(f"PPE Detector started with {MODEL}! Press Q to quit.")

# ── Trackers ─────────────────────────────────────
last_logged     = {}
last_alert_time = 0
fps_start_time  = time.time()
frame_count     = 0
fps             = 0

# ── Main Loop ─────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    frame_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count    = 0
        fps_start_time = time.time()

    # Run detection
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
    annotated_frame = results[0].plot()

    # Check violations
    violation_found = False
    current_time    = datetime.now()
    now             = time.time()

    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            conf     = float(box.conf)

            if cls_name in VIOLATION_CLASSES:
                violation_found = True

                # Log with cooldown
                if cls_name not in last_logged or \
                   (now - last_logged[cls_name]) > LOG_COOLDOWN:
                    csv_writer.writerow([
                        current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        cls_name,
                        f"{conf:.2f}",
                        MODEL
                    ])
                    csv_file.flush()
                    last_logged[cls_name] = now
                    print(f"[{current_time.strftime('%H:%M:%S')}] Violation: {cls_name} ({conf:.2f})")

    # ── Overlay UI ───────────────────────────────
    # Violation status
    if violation_found:
        cv2.putText(annotated_frame, "WARNING: VIOLATION DETECTED!",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        if now - last_alert_time > ALERT_COOLDOWN:
            winsound.Beep(1000, 500)
            last_alert_time = now
    else:
        cv2.putText(annotated_frame, "ALL CLEAR",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    # FPS counter
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    # Model name
    cv2.putText(annotated_frame, f"Model: {MODEL}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    # Timestamp
    cv2.putText(annotated_frame,
                current_time.strftime("%Y-%m-%d %H:%M:%S"),
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    cv2.imshow("PPE Detector - Press Q to quit", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("Detector stopped. Check logs/violations_log.csv for records.")