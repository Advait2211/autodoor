import torch
import cv2
import easyocr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from pathlib import Path

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Perform OCR on bounding box region
def ocr_image(img, coordinates):
    x1, y1, x2, y2 = map(int, coordinates)
    cropped = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)

    text = ""
    for res in results:
        if len(results) == 1:
            text = res[1]
        if len(results) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
    return str(text)

# Main function
def predict(source, model_path="yolov8n.pt", save_dir="runs/ocr"):
    model = YOLO(model_path)
    results = model(source, save=True, save_txt=True, conf=0.25)

    for r_idx, r in enumerate(results):
        img = r.orig_img.copy()
        annotator = Annotator(img)

        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            name = model.names[cls]

            text = ocr_image(img, xyxy)
            label = f"{text or name} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(cls, True))

        # Save the image with OCR results
        save_path = Path(save_dir) / source
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotator.result())

if __name__ == "__main__":
    for i in range(1, 12):
        predict(source=f"Dataset/{i}.jpeg")

