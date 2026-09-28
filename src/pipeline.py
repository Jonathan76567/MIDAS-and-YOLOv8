import os

import cv2
import numpy as np

from src.cost_calculator import CostCalculator
from src.depth_estimator import DepthEstimator
from src.detector import SegDetector
from src import panel_matcher

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Weight hasil training dipakai otomatis bila sudah ada; sebelum itu pakai placeholder
DAMAGE_WEIGHTS = os.path.join(ROOT_DIR, 'weights', 'damage_best.pt')
PARTS_WEIGHTS = os.path.join(ROOT_DIR, 'weights', 'parts_best.pt')
PLACEHOLDER_WEIGHTS = os.path.join(ROOT_DIR, 'yolov8s-seg.pt')
PRICE_LIST = os.path.join(ROOT_DIR, 'data', 'price_list.json')


def _existing(path):
    return path if os.path.exists(path) and os.path.getsize(path) > 0 else None


class BodyRepairPipeline:
    def __init__(self, damage_model_path=None, parts_model_path=None,
                 price_list_path=PRICE_LIST, conf_threshold=0.25, midas_type='MiDaS_small'):
        """
        Pipeline lengkap: deteksi kerusakan -> deteksi panel -> pencocokan
        -> estimasi kedalaman -> kalkulasi biaya.

        - damage_model_path: default weights/damage_best.pt, fallback ke yolov8s-seg.pt
        - parts_model_path: default weights/parts_best.pt; jika belum ada, deteksi panel
          dinonaktifkan dan panel dipilih manual oleh pengguna
        """
        damage_model_path = damage_model_path or _existing(DAMAGE_WEIGHTS) or PLACEHOLDER_WEIGHTS
        parts_model_path = parts_model_path or _existing(PARTS_WEIGHTS)

        self.using_placeholder = damage_model_path == PLACEHOLDER_WEIGHTS
        self.damage_detector = SegDetector(damage_model_path, conf_threshold)
        self.parts_detector = SegDetector(parts_model_path, conf_threshold) if parts_model_path else None
        self.depth_estimator = DepthEstimator(midas_type)
        self.calculator = CostCalculator(price_list_path)

    # ------------------------------------------------------------------
    # Tahap 1: inferensi semua model (berat, hasilnya bisa di-cache)
    # ------------------------------------------------------------------
    def run_models(self, image_bgr, conf=None):
        damage_dets, _ = self.damage_detector.predict(image_bgr, conf=conf)
        part_dets = []
        if self.parts_detector is not None:
            part_dets, _ = self.parts_detector.predict(image_bgr, conf=conf)
        depth_map = self.depth_estimator.estimate(image_bgr)

        return {
            "image": image_bgr,
            "damage_dets": damage_dets,
            "part_dets": part_dets,
            "depth_map": depth_map,
        }

    # ------------------------------------------------------------------
    # Tahap 2: pencocokan panel + kalkulasi biaya (ringan)
    # ------------------------------------------------------------------
    def estimate(self, raw, manual_panel=None):
        """
        - manual_panel: key panel pilihan pengguna untuk kerusakan yang tidak
          menempel ke panel terdeteksi. Untuk kelompok ini A_mask dihitung
          terhadap seluruh gambar (asumsi foto close-up satu panel).
        """
        image = raw["image"]
        depth_map = raw["depth_map"]

        panels, unassigned = panel_matcher.match(
            raw["damage_dets"], raw["part_dets"], self.calculator.panel_mapping
        )

        if manual_panel and unassigned:
            union = np.zeros(image.shape[:2], dtype=bool)
            for d in unassigned:
                union |= d["mask_binary"].astype(bool)
            panels.append({
                "panel_key": manual_panel,
                "part_class": "(dipilih manual)",
                "part_mask": None,
                "damage_mask": union,
                "damages": unassigned,
                "mask_area_ratio": float(union.sum() / union.size),
            })
            unassigned = []

        items = []
        for p in panels:
            d_depth = self._panel_depth(depth_map, p)
            if p["panel_key"] is None:
                cost, status = None, f"Harga panel '{p['part_class']}' belum tersedia"
            else:
                cost, status = self.calculator.calculate_cost(p["panel_key"], p["mask_area_ratio"], d_depth)
            items.append({
                "panel_key": p["panel_key"],
                "part_class": p["part_class"],
                "damage_classes": sorted({d["class_name"] for d in p["damages"]}),
                "n_damages": len(p["damages"]),
                "mask_area_ratio": p["mask_area_ratio"],
                "depth_deviation": d_depth,
                "cost": cost,
                "status": status,
            })

        total = sum(i["cost"]["total_estimated_cost"] for i in items if i["cost"])

        return {
            "items": items,
            "unassigned": unassigned,
            "total": total,
            "formatted_total": f"Rp {total:,.0f}".replace(",", "."),
            "overlay": self.draw_overlay(image, raw["part_dets"], raw["damage_dets"]),
            "depth_vis": DepthEstimator.colorize(depth_map),
        }

    def _panel_depth(self, depth_map, panel):
        """
        D_depth hanya dari kerusakan yang punya dimensi kedalaman (price_list:
        kelas_kedalaman, mis. Dent). Goresan/cat mengelupas tidak diukur agar
        MiDaS tidak memberi kedalaman palsu. Tanpa kerusakan tersebut, D_depth = 0.
        """
        depth_dmgs = [d for d in panel["damages"] if d["class_name"] in self.calculator.depth_classes]
        if not depth_dmgs:
            return 0.0

        mask = np.zeros(depth_map.shape, dtype=bool)
        for d in depth_dmgs:
            mask |= d["mask_binary"].astype(bool)
        if panel["part_mask"] is not None:
            mask &= panel["part_mask"]
        return DepthEstimator.depth_deviation(depth_map, mask, panel["part_mask"])

    def analyze(self, image_bgr, manual_panel=None, conf=None):
        return self.estimate(self.run_models(image_bgr, conf=conf), manual_panel)

    def draw_overlay(self, image, part_dets, damage_dets):
        """Panel: biru transparan + label. Kerusakan: merah transparan + kontur + label."""
        out = image.copy()
        mapping = self.calculator.panel_mapping

        layer = out.copy()
        for p in part_dets:
            if p["class_name"] in mapping:
                layer[p["mask_binary"].astype(bool)] = (255, 140, 0)
        out = cv2.addWeighted(layer, 0.25, out, 0.75, 0)

        layer = out.copy()
        for d in damage_dets:
            layer[d["mask_binary"].astype(bool)] = (0, 0, 255)
        out = cv2.addWeighted(layer, 0.45, out, 0.55, 0)

        for d in damage_dets:
            contours, _ = cv2.findContours(d["mask_binary"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

        for p in part_dets:
            if p["class_name"] in mapping:
                x1, y1 = p["bbox"][:2]
                self._label(out, mapping[p["class_name"]] or p["class_name"], (x1, y1 + 18), (255, 140, 0))
        for d in damage_dets:
            x1, y1 = d["bbox"][:2]
            self._label(out, f'{d["class_name"]} {d["confidence"]:.2f}', (x1, max(y1 - 4, 14)), (0, 0, 255))

        return out

    @staticmethod
    def _label(img, text, org, color):
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = org
        cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), color, -1)
        cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
