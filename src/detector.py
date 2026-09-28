import cv2
import numpy as np
from ultralytics import YOLO


class SegDetector:
    def __init__(self, model_path='yolov8s-seg.pt', conf_threshold=0.25):
        """
        Wrapper generik YOLOv8-Seg. Dipakai dua kali di pipeline:
        - model kerusakan (dent, scratch, crack, ...)
        - model panel mobil (front_door, hood, bumper, ...)

        Param:
        - model_path: path ke pretrained weight atau weight hasil training (best.pt)
        - conf_threshold: batas minimal tingkat kepercayaan deteksi
        """
        print(f"[INFO] Memuat model YOLOv8-Seg dari {model_path}...")
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def predict(self, image_input, conf=None):
        """
        Melakukan inferensi segmentasi pada gambar.

        Param:
        - image_input: path string ke file gambar ATAU cv2 image array BGR (numpy array)
        - conf: override confidence threshold (opsional)

        Returns:
        - detections: list of dictionary berisi hasil deteksi per objek
        - annotated_img: gambar hasil visualisasi bawaan YOLO
        """
        # Membaca gambar jika input berupa path string
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_input}")
        else:
            image = image_input.copy()

        img_h, img_w = image.shape[:2]
        total_img_pixels = img_h * img_w

        # Jalankan inferensi YOLO
        conf = self.conf_threshold if conf is None else conf
        results = self.model(image, conf=conf, verbose=False)[0]

        detections = []
        annotated_img = results.plot()

        # Cek apakah ada objek/mask yang terdeteksi
        if results.masks is not None and len(results.masks) > 0:
            masks = results.masks.data.cpu().numpy()  # Binary masks (N, H, W)
            boxes = results.boxes

            for i, mask in enumerate(masks):
                # Resize mask ke ukuran asli gambar
                mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                mask_pixel_count = int(np.sum(mask_binary))
                if mask_pixel_count == 0:
                    continue

                # Ambil info kelas dan confidence
                class_id = int(boxes.cls[i].item())
                class_name = self.model.names[class_id]
                confidence = float(boxes.conf[i].item())

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [int(v) for v in boxes.xyxy[i].tolist()],
                    "mask_binary": mask_binary,
                    "mask_pixel_count": mask_pixel_count,
                    # Rasio terhadap seluruh gambar (dipakai hanya sebagai fallback
                    # bila panel tidak terdeteksi; rasio utama dihitung terhadap panel)
                    "image_area_ratio": round(mask_pixel_count / total_img_pixels, 4)
                })

        return detections, annotated_img


# Alias agar kode lama tetap jalan
DamageDetector = SegDetector


# --- Pengujian Mandiri Detector ---
if __name__ == "__main__":
    detector = SegDetector(model_path='yolov8s-seg.pt', conf_threshold=0.25)

    test_img = 'bus.jpg'
    print("[INFO] Menguji inferensi pada gambar sampel...")
    results, output_img = detector.predict(test_img)

    print(f"\nTotal objek terdeteksi: {len(results)}")
    for idx, det in enumerate(results):
        print(f"\n--- Objek #{idx+1} ---")
        print(f"Kelas        : {det['class_name']}")
        print(f"Confidence   : {det['confidence']}")
        print(f"Rasio Luas   : {det['image_area_ratio']} ({det['image_area_ratio']*100:.2f}% dari gambar)")

    cv2.imwrite("test_detection_output.jpg", output_img)
    print("\n[INFO] Gambar hasil deteksi disimpan sebagai 'test_detection_output.jpg'")
