import sys

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_type='MiDaS_small'):
        """
        Estimasi kedalaman monokular menggunakan MiDaS (intel-isl/MiDaS via torch.hub).
        - model_type: 'MiDaS_small' (ringan, cocok untuk CPU), 'DPT_Hybrid', atau 'DPT_Large'

        Catatan: output MiDaS adalah *relative inverse depth* (nilai besar = lebih dekat
        ke kamera), bukan jarak dalam satuan meter. Karena itu kedalaman penyok diukur
        sebagai deviasi relatif terhadap permukaan di sekitarnya.
        """
        print(f"[INFO] Memuat model MiDaS ({model_type})...")
        # MiDaS_small memuat backbone dari repo ini; tandai trusted agar tidak ada prompt input
        torch.hub.list('rwightman/gen-efficientnet-pytorch', trust_repo=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('intel-isl/MiDaS', model_type, trust_repo=True)
        self.model.to(self.device).eval()

        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        if model_type in ('DPT_Large', 'DPT_Hybrid'):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def estimate(self, image_bgr):
        """
        Menghasilkan depth map (float32, ukuran sama dengan gambar input).
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)

    @staticmethod
    def depth_deviation(depth_map, damage_mask, region_mask=None):
        """
        Menghitung deviasi kedalaman area kerusakan (D_depth, rentang 0.0 - 1.0).

        D_depth = |median(depth di dalam mask) - median(depth cincin sekitar mask)|
                  / rentang depth pada region (persentil 98 - persentil 2)

        Param:
        - depth_map: output estimate()
        - damage_mask: binary mask kerusakan (H, W)
        - region_mask: binary mask panel (opsional). Jika ada, cincin pembanding dan
          normalisasi dibatasi di dalam panel agar tidak terpengaruh latar belakang.
        """
        damage = damage_mask.astype(bool)
        if damage.sum() == 0:
            return 0.0

        # Cincin di sekitar kerusakan = hasil dilasi mask dikurangi mask itu sendiri
        k = max(7, int(0.15 * np.sqrt(damage.sum())))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(damage.astype(np.uint8), kernel).astype(bool)
        ring = dilated & ~damage

        if region_mask is not None:
            region = region_mask.astype(bool) | damage
            ring_in_region = ring & region
            if ring_in_region.sum() > 0:
                ring = ring_in_region
        else:
            region = dilated

        if ring.sum() == 0:
            return 0.0

        d_inside = np.median(depth_map[damage])
        d_ring = np.median(depth_map[ring])

        region_vals = depth_map[region]
        depth_range = np.percentile(region_vals, 98) - np.percentile(region_vals, 2)
        if depth_range <= 1e-6:
            return 0.0

        deviation = abs(d_inside - d_ring) / depth_range
        return float(np.clip(deviation, 0.0, 1.0))

    @staticmethod
    def colorize(depth_map):
        """Mengubah depth map menjadi heatmap BGR untuk visualisasi."""
        d = depth_map - depth_map.min()
        d = (255 * d / (d.max() + 1e-6)).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


# --- Pengujian Mandiri Depth Estimator ---
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'bus.jpg'
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {img_path}")

    estimator = DepthEstimator()
    depth = estimator.estimate(image)
    print(f"Depth map: shape={depth.shape}, min={depth.min():.2f}, max={depth.max():.2f}")

    cv2.imwrite("test_depth_output.jpg", DepthEstimator.colorize(depth))
    print("[INFO] Heatmap kedalaman disimpan sebagai 'test_depth_output.jpg'")
