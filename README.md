\# PPE Safety Compliance Detection System 🦺



Real-time Personal Protective Equipment (PPE) detection system using YOLOv8 and NVIDIA Jetson Nano for construction site safety monitoring.



\---



\## Problem Statement

Construction sites require workers to wear PPE (helmets, vests, masks, gloves) at all times. Manual monitoring is inefficient and error-prone. This system automates PPE compliance detection using AI-powered computer vision, providing real-time alerts when violations are detected.



\---



\## Role of Edge Computing

| Component | Runs On |

|---|---|

| YOLOv8n inference | NVIDIA Jetson Nano |

| Real-time camera feed | Jetson Nano |

| Violation logging | Jetson Nano (local CSV) |

| Alert system | Jetson Nano (buzzer) |



\*\*Why Edge?\*\*

\- No internet required

\- Low latency (<40ms)

\- Data privacy preserved

\- One-time hardware cost



\---



\## Methodology

Camera Feed → Preprocessing → YOLOv8n Inference → Violation Check → Alert + Log

\---



\## Model Details

| Property | YOLOv8n | YOLOv8s |

|---|---|---|

| Parameters | 3.01M | 11.13M |

| Model Size | 6.2 MB | 22.5 MB |

| mAP50 | 50.3% | 58.7% |

| Inference (CPU) | 82.6ms | 164.9ms |

| FPS (CPU) | 12.1 | 6.1 |

| Selected for Jetson | ✅ Yes | ❌ No |



\- \*\*Framework:\*\* PyTorch + Ultralytics

\- \*\*Input Size:\*\* 640×640

\- \*\*Classes:\*\* 25 (including 7 PPE-specific)



\---



\## Training Details

\- \*\*Dataset:\*\* Construction Site Safety (Roboflow Universe)

\- \*\*Images:\*\* 3,000+ labeled images

\- \*\*Classes:\*\* 25 (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Mask, NO-Mask, Gloves, etc.)

\- \*\*Epochs:\*\* 44 (YOLOv8n), 50 (YOLOv8s)

\- \*\*Device:\*\* CPU (Intel Core i7-1255U)



\### Key Results

| Class | mAP50 (n) | mAP50 (s) |

|---|---|---|

| Hardhat | 70.3% | 74.9% |

| NO-Hardhat | 53.9% | 59.4% |

| Mask | 83.7% | 86.0% |

| NO-Mask | 39.9% | 51.7% |

| Safety Vest | 70.9% | 74.5% |

| NO-Safety Vest | 59.4% | 63.2% |



\---



\## Project Structure



ppe\_project/

├── src/

│   ├── main.py           # Main entry point

│   ├── config.py         # Configuration settings

│   ├── inference.py      # Model inference pipeline

│   ├── preprocessing.py  # Frame preparation

│   ├── logger.py         # Violation CSV logger

│   ├── utils.py          # Helper functions

│   ├── training.py       # Model training script

│   └── benchmark.py      # Speed benchmarking

├── models/

│   ├── yolov8n/best.pt   # Trained YOLOv8n weights

│   └── yolov8s/best.pt   # Trained YOLOv8s weights

├── dataset/              # Construction site dataset

├── logs/                 # Violation CSV logs

├── results/              # Training graphs \& results

└── requirements.txt



\---



\## Setup Instructions



\### 1. Clone Repository

```bash

git clone https://github.com/YOUR\_USERNAME/ppe-safety-detector.git

cd ppe-safety-detector

```



\### 2. Create Environment

```bash

conda create -n ppe\_detector python=3.10 -y

conda activate ppe\_detector

```



\### 3. Install Dependencies

```bash

pip install -r requirements.txt

```



\### 4. Run Detection

```bash

cd src

python main.py

```



\### 5. Switch Model

In `src/config.py`:

```python

MODEL\_NAME = "yolov8n"  # or "yolov8s"

INPUT\_SOURCE = 0        # 0=webcam, "path/video.mp4"=video

```



\---



\## Results

\- Real-time detection at \*\*12 FPS\*\* on CPU

\- \*\*30+ FPS\*\* expected on Jetson Nano with TensorRT

\- Sound alert on violation detection

\- Automatic CSV violation logging with cooldown



\---



\## Tech Stack

\- Python 3.10

\- YOLOv8 (Ultralytics)

\- OpenCV

\- PyTorch

\- NVIDIA Jetson Nano

\- TensorRT (planned)



\---



\## Future Scope

\- Person Re-ID for cross-camera tracking

\- Telegram alert with violation photo

\- Web dashboard for remote monitoring

\- TensorRT optimization on Jetson Nano



\---



\## References

\- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics

\- Dataset: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

\- NVIDIA Jetson Nano: https://developer.nvidia.com/embedded/jetson-nano

