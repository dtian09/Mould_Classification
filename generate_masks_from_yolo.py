import os
import numpy as np
from PIL import Image

def yolo_to_bbox(yolo_line, img_w, img_h):
    # YOLO format: class cx cy w h (all normalized)
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        return None
    _, cx, cy, w, h = map(float, parts)
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return x1, y1, x2, y2

def generate_masks(image_dir, label_dir, mask_dir, image_exts=['.jpg', '.jpeg', '.png', '.webp']):
    os.makedirs(mask_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        img_base = label_file[:-4]
        # Find corresponding image
        img_path = None
        for ext in image_exts:
            candidate = os.path.join(image_dir, img_base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"Image for {label_file} not found.")
            continue
        img = Image.open(img_path)
        img_w, img_h = img.size
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                bbox = yolo_to_bbox(line, img_w, img_h)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    mask[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = 255
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(mask_dir, img_base + '.png'))
        print(f"Saved mask for {img_base}")

if __name__ == "__main__":
    #image_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/train/images"
    #label_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/train/labels"
    #mask_dir = "segmentation/masks"
    #image_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/test/images"
    #label_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/test/labels"
    #mask_dir = "segmentation/masks_test"
    image_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/valid/images"
    label_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/valid/labels"
    mask_dir = "segmentation/masks_valid"
    generate_masks(image_dir, label_dir, mask_dir)
