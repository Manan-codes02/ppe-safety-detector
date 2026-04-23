from ultralytics import YOLO
import time
import cv2

# Test YOLOv8n
print("Testing YOLOv8n...")
model_n = YOLO(r"C:\Users\Manan Singla\Desktop\ppe_project\models\yolov8n\best.pt")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

times = []
for i in range(10):
    start = time.time()
    model_n.predict(frame, conf=0.35, verbose=False)
    times.append((time.time()-start)*1000)
print(f"YOLOv8n avg inference: {sum(times)/len(times):.1f}ms")
print(f"YOLOv8n FPS: {1000/(sum(times)/len(times)):.1f}")

# Test YOLOv8s
print("\nTesting YOLOv8s...")
model_s = YOLO(r"C:\Users\Manan Singla\Desktop\ppe_project\models\yolov8s\best.pt")
times = []
for i in range(10):
    start = time.time()
    model_s.predict(frame, conf=0.35, verbose=False)
    times.append((time.time()-start)*1000)
print(f"YOLOv8s avg inference: {sum(times)/len(times):.1f}ms")
print(f"YOLOv8s FPS: {1000/(sum(times)/len(times)):.1f}")