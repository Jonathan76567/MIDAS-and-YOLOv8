import sys

import cv2
import numpy as np
from ultralytics import YOLO


class SegDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        print(f"[INFO] Memuat model YOLOv8-Seg dari {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def predict(self, image, conf=None):
        img_h, img_w = image.shape[:2]
        conf = self.conf_threshold if conf is None else conf
        results = self.model(image, conf=conf, verbose=False)[0]

        detections = []
        if results.masks is None:
            return detections

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes
        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            if mask_binary.sum() == 0:
                continue

            detections.append({
                "class_name": self.model.names[int(boxes.cls[i].item())],
                "confidence": round(float(boxes.conf[i].item()), 3),
                "bbox": [int(v) for v in boxes.xyxy[i].tolist()],
                "mask_binary": mask_binary,
            })

        return detections


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Pemakaian: python -m src.detector <path_weight> <path_foto>")
    img = cv2.imread(sys.argv[2])
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {sys.argv[2]}")

    dets = SegDetector(sys.argv[1]).predict(img)
    print(f"\nTotal objek terdeteksi: {len(dets)}")
    for idx, det in enumerate(dets, 1):
        area = det['mask_binary'].sum() / det['mask_binary'].size
        print(f"#{idx} {det['class_name']:15s} confidence {det['confidence']:.3f} | luas {area * 100:.2f}% gambar")
