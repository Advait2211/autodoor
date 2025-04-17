import cv2
import torch
import re
import easyocr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

# Regex pattern for filtering out valid number plate-like text
def is_valid_plate(text):
    return bool(re.match(r'^[A-Z0-9\-]{6,12}$', text.strip().upper()))

# Preprocess cropped image before OCR
def preprocess_plate(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Perform OCR on the number plate region
def read_plate_text(cropped):
    processed = preprocess_plate(cropped)
    result = reader.readtext(processed)

    for res in result:
        text = res[1].strip().upper()
        if is_valid_plate(text):
            return text
    return None

# Main pipeline: image → detect plates → OCR → print plate number
def detect_plate_number(image_path, model_path="keremberke/yolov8n-license-plate"):
    model = YOLO(model_path)
    results = model(image_path)

    for r in results:
        img = r.orig_img.copy()
        annotator = Annotator(img)
        found_plate = False

        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            cls = int(box.cls[0])

            # Crop and OCR
            x1, y1, x2, y2 = map(int, xyxy)
            cropped = img[y1:y2, x1:x2]
            plate_text = read_plate_text(cropped)

            if plate_text:
                found_plate = True
                print("Detected Number Plate:", plate_text)
                annotator.box_label(xyxy, plate_text, color=colors(cls, True))

        if not found_plate:
            print("No valid plate detected.")

        # Optional: Save annotated image
        cv2.imwrite(f'./results/{image_path}', annotator.result())

if __name__ == "__main__":
    for i in range(1, 12):
        detect_plate_number(image_path=f"Dataset/{i}.jpeg")
