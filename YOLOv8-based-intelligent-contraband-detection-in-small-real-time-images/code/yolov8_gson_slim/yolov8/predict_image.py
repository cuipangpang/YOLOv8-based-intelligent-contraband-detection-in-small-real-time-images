from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.predict('ultralytics/assets', save=True)
