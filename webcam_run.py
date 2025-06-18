# 📦 Install dependencies before running:
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# 🧠 Load your trained YOLOv8 model
model_path = 'C:\\Users\HP\Desktop\Yolov8n combined\\weights\\best.pt'  # ✅ Replace this with the path to your YOLOv8 model
model = YOLO(model_path)

# 🎥 Start webcam capture (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

print("✅ Running YOLOv8 on webcam... Press 'q' to quit.")

while True:
    # 🔄 Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # 🔍 Run inference
    results = model(frame)

    # 🖼️ Plot predictions on the frame
    annotated_frame = results[0].plot()

    # 📺 Show the frame
    cv2.imshow("YOLOv8 Live Inference", annotated_frame)

    # ⏹️ Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Clean up
cap.release()
cv2.destroyAllWindows()
print("✅ Inference ended. Webcam released.")
import psutil

# CPU usage
print("CPU Usage (%):", psutil.cpu_percent(interval=1))

# RAM usage
mem = psutil.virtual_memory()
print("RAM Used:", mem.used / 1024**2, "MB")
print("RAM Available:", mem.available / 1024**2, "MB")