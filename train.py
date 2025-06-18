from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Choose from yolov8n.pt, yolov8s.pt, etc.

# Train the model
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=50,
    imgsz=640,
    patience=10,  # Optional: stop if no improvement for 10 epochs
    save=True,
    project='/kaggle/working/yolo_project',
    name='yolov8_run',
    verbose=True
)