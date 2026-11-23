import sys

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'


class DepthEstimator:
    def __init__(self, model_id=MODEL_ID):
        print(f"[INFO] Memuat model kedalaman ({model_id})...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self._crop_cache = {}

    def estimate(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            inputs = self.processor(images=img_rgb, return_tensors='pt').to(self.device)
            prediction = self.model(**inputs).predicted_depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)

    @staticmethod
    def surface_deviation(depth_map, damage_mask):
        damage = damage_mask.astype(bool)
        if damage.sum() == 0:
            return 0.0
        k = max(7, int(0.15 * np.sqrt(damage.sum())))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ring = cv2.dilate(damage.astype(np.uint8), kernel).astype(bool) & ~damage
        if ring.sum() < 10:
            return 0.0

        h, w = depth_map.shape

        # permukaan kuadratik z = f(x, y)
        def design(xs, ys):
            xs, ys = xs / w, ys / h
            return np.stack([xs, ys, np.ones_like(xs), xs * xs, ys * ys, xs * ys], 1)

        ys, xs = np.nonzero(ring)
        coef, *_ = np.linalg.lstsq(design(xs, ys), depth_map[ring].astype(np.float64), rcond=None)
        yi, xi = np.nonzero(damage)
        resid = depth_map[damage] - design(xi, yi) @ coef

        vals = np.r_[depth_map[damage], depth_map[ring]]
        span = np.percentile(vals, 98) - np.percentile(vals, 2)
        if span <= 1e-6:
            return 0.0
        return float(np.clip(np.percentile(np.abs(resid), 75) / span, 0.0, 1.0))

    def crop_depth_deviation(self, image_bgr, damage_mask, margin=0.25):
        mask = damage_mask.astype(np.uint8)
        if mask.sum() == 0:
            return 0.0
        x, y, bw, bh = cv2.boundingRect(mask)
        h, w = mask.shape
        px, py = int(bw * margin) + 8, int(bh * margin) + 8
        x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(w, x + bw + px), min(h, y + bh + py)

        crop = image_bgr[y0:y1, x0:x1]
        key = (x0, y0, x1, y1, hash(crop.tobytes()))
        if key not in self._crop_cache:
            if len(self._crop_cache) > 256:
                self._crop_cache.clear()
            self._crop_cache[key] = self.estimate(crop)
        return self.surface_deviation(self._crop_cache[key], mask[y0:y1, x0:x1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Pemakaian: python -m src.depth_estimator <path_foto>")
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {sys.argv[1]}")

    depth = DepthEstimator().estimate(image)
    print(f"Depth map: shape={depth.shape}, min={depth.min():.2f}, max={depth.max():.2f}")
