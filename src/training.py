# ─────────────────────────────────────────────
# training.py — Model Training Script
# PPE Safety Compliance Detection System
# ─────────────────────────────────────────────

import os
from ultralytics import YOLO
from config import BASE_DIR, RESULTS_DIR, IMAGE_SIZE


# ── Dataset Config ────────────────────────────
DATASET_YAML = os.path.join(
    BASE_DIR, "dataset",
    "Construction-Site-Safety-30",
    "data.yaml"
)

# ── Training Hyperparameters ──────────────────
HYPERPARAMS = {
    # Core settings
    "data"     : DATASET_YAML,
    "epochs"   : 50,
    "imgsz"    : IMAGE_SIZE,
    "batch"    : 4,
    "device"   : "cpu",
    "workers"  : 2,

    # Output settings
    "project"  : RESULTS_DIR,
    "save"     : True,
    "plots"    : True,

    # Early stopping
    "patience" : 10,

    # Optimization
    "optimizer": "SGD",
    "lr0"      : 0.01,
    "momentum" : 0.937,
    "weight_decay": 0.0005,

    # Augmentation
    "mosaic"   : 1.0,
    "flipud"   : 0.0,
    "fliplr"   : 0.5,
}


def train_model(model_name="yolov8n"):
    """
    Train YOLOv8 model on PPE dataset.

    Args:
        model_name: "yolov8n" or "yolov8s"

    Training Steps:
    1. Load pretrained base model
    2. Configure hyperparameters
    3. Train on dataset
    4. Save best weights
    5. Generate training graphs
    """

    print("=" * 50)
    print(f"  Training: {model_name}")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"  Epochs  : {HYPERPARAMS['epochs']}")
    print(f"  Device  : {HYPERPARAMS['device']}")
    print("=" * 50)

    # ── Load Base Model ───────────────────────
    print(f"\n[Training] Loading base model: {model_name}.pt")
    model = YOLO(f"{model_name}.pt")

    # ── Set output folder name ─────────────────
    run_name = f"ppe_model_{model_name}"
    HYPERPARAMS["name"] = run_name

    # ── Start Training ────────────────────────
    print("[Training] Starting training...\n")
    results = model.train(**HYPERPARAMS)

    # ── Print Results ─────────────────────────
    print("\n" + "=" * 50)
    print("  Training Complete!")
    print("=" * 50)
    print(f"  Best model : {results.save_dir}\\weights\\best.pt")
    print(f"  mAP50      : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    import sys

    # Allow passing model name as argument
    # Usage: python training.py yolov8n
    #        python training.py yolov8s
    model = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
    train_model(model)