from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model
model.train(data="config.yaml", epochs=1)  # Train the model on your dataset
