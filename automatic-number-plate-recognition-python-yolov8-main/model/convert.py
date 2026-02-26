import os
from PIL import Image

LABEL_DIR = "/Users/advaitdesai/Programming/autodoor/automatic-number-plate-recognition-python-yolov8-main/model/data/labels/train"
IMG_DIR = "/Users/advaitdesai/Programming/autodoor/automatic-number-plate-recognition-python-yolov8-main/model/data/images/train"

for filename in os.listdir(LABEL_DIR):
    if not filename.endswith(".txt"):
        continue

    label_path = os.path.join(LABEL_DIR, filename)
    image_filename = os.path.splitext(filename)[0] + ".jpg"
    image_path = os.path.join(IMG_DIR, image_filename)

    # Skip if image doesn't exist
    if not os.path.exists(image_path):
        print(f"[SKIP] Image not found for label: {filename}")
        continue

    # Get actual image size
    with Image.open(image_path) as img:
        IMG_WIDTH, IMG_HEIGHT = img.size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()

        if parts[0] == "0":
            new_lines.append(line.strip())
            continue

        if parts[:3] == ["Vehicle", "registration", "plate"] and len(parts) == 7:
            try:
                x1, y1, x2, y2 = map(float, parts[3:])
                cx = ((x1 + x2) / 2) / IMG_WIDTH
                cy = ((y1 + y2) / 2) / IMG_HEIGHT
                w = abs(x2 - x1) / IMG_WIDTH
                h = abs(y2 - y1) / IMG_HEIGHT
                # Clamp values to 1.0 max to be safe
                cx = min(cx, 1.0)
                cy = min(cy, 1.0)
                w = min(w, 1.0)
                h = min(h, 1.0)
                new_line = f"0 {cx:.16f} {cy:.16f} {w:.16f} {h:.16f}"
                new_lines.append(new_line)
            except Exception as e:
                print(f"[ERROR] in {filename}: {e}")
        else:
            print(f"[WARNING] Skipped unknown line format in {filename}: {line.strip()}")

    with open(label_path, 'w') as f:
        for nl in new_lines:
            f.write(nl + '\n')
