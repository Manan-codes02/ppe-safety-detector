from ultralytics import YOLO

model = YOLO(r"C:\Users\Manan Singla\Desktop\ppe_project\models\yolov8n\best.pt")

results = model.predict(
    source=r"C:\Users\Manan Singla\Desktop\ppe_project\dataset\Construction-Site-Safety-30\test\images",
    conf=0.35,
    save=True,
    project=r"C:\Users\Manan Singla\Desktop\ppe_project\results",
    name="test_results"
)

print("Testing complete! Check test_results folder for output images.")