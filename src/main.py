# ─────────────────────────────────────────────
# main.py — Main Entry Point
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import cv2
import sys
from config import INPUT_SOURCE, WINDOW_TITLE
from preprocessing import preprocess_frame, open_source
from inference import PPEDetector
from logger import ViolationLogger
from utils import FPSCounter, draw_overlay, AlertManager, print_system_info


def main():
    # ── System Info ───────────────────────────
    print_system_info()

    # ── Initialize Components ─────────────────
    detector   = PPEDetector()       # loads model
    logger     = ViolationLogger()   # opens CSV
    fps_counter = FPSCounter()       # tracks FPS
    alert       = AlertManager()     # manages beeps

    # ── Open Input Source ─────────────────────
    try:
        cap = open_source(INPUT_SOURCE)
    except ValueError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    print("\n[Main] System started! Press Q or ESC to quit.\n")

    # ── Main Detection Loop ───────────────────
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[Main] No frame received. Stopping.")
            break

        # Step 1 — Preprocess frame
        frame = preprocess_frame(frame)

        # Step 2 — Run inference
        annotated_frame, violations, inference_ms = detector.run(frame)

        # Step 3 — Update FPS
        fps = fps_counter.update()

        # Step 4 — Log violations
        if violations:
            logger.log(violations)
            alert.trigger()

        # Step 5 — Draw UI overlay
        violation_found = len(violations) > 0
        annotated_frame = draw_overlay(
            annotated_frame, fps, violation_found
        )

        # Step 6 — Show frame
        cv2.imshow(WINDOW_TITLE, annotated_frame)

        # Step 7 — Check quit keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n[Main] Quit key pressed.")
            break

    # ── Cleanup ───────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    detector.print_stats()
    print("[Main] System stopped cleanly.")


if __name__ == "__main__":
    main()