import os
import cv2

LABEL_DIR = "/Users/advaitdesai/Programming/autodoor/automatic-number-plate-recognition-python-yolov8-main/model/data/labels/train"
IMG_DIR = "/Users/advaitdesai/Programming/autodoor/automatic-number-plate-recognition-python-yolov8-main/model/data/images/train"

def draw_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        _, cx, cy, bw, bh = map(float, parts)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('YOLO Labels', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Loop through first 50 images
shown = 0
for filename in os.listdir(IMG_DIR):
    if not filename.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt"))

    if os.path.exists(label_path):
        print(f"Showing {filename} ({shown+1}/50)...")
        draw_boxes(img_path, label_path)
        shown += 1

    if shown >= 50:
        break
