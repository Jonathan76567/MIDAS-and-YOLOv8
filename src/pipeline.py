import os

import cv2
import numpy as np

from src.cost_calculator import CostCalculator
from src.depth_estimator import DepthEstimator
from src.detector import SegDetector
from src import panel_matcher

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DAMAGE_WEIGHTS = os.path.join(ROOT_DIR, 'weights', 'damage_best.pt')
PARTS_WEIGHTS = os.path.join(ROOT_DIR, 'weights', 'parts_best.pt')
PRICE_LIST = os.path.join(ROOT_DIR, 'data', 'price_list.json')

DEPTH_CROP_MARGIN = 0.25

# threshold kerusakan hasil tuning di data validasi
DAMAGE_CONF = 0.10
PARTS_CONF = 0.25


class BodyRepairPipeline:
    def __init__(self, damage_model_path=DAMAGE_WEIGHTS, parts_model_path=PARTS_WEIGHTS,
                 price_list_path=PRICE_LIST, damage_conf=DAMAGE_CONF, parts_conf=PARTS_CONF):
        for path in (damage_model_path, parts_model_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model tidak ditemukan: {path}. Latih dulu dengan train.py")

        self.damage_detector = SegDetector(damage_model_path, damage_conf)
        self.parts_detector = SegDetector(parts_model_path, parts_conf)
        self.depth_estimator = DepthEstimator()
        self.calculator = CostCalculator(price_list_path)

    def run_models(self, image_bgr, damage_conf=None, parts_conf=None):
        return {
            "image": image_bgr,
            "damage_dets": self.damage_detector.predict(image_bgr, conf=damage_conf),
            "part_dets": self.parts_detector.predict(image_bgr, conf=parts_conf),
        }

    def estimate(self, raw, manual_panel=None):
        image = raw["image"]

        panels, unassigned = panel_matcher.match(
            raw["damage_dets"], raw["part_dets"], self.calculator.panel_mapping
        )

        # kerusakan tanpa panel: luas dihitung terhadap seluruh gambar
        if manual_panel and unassigned:
            union = np.zeros(image.shape[:2], dtype=bool)
            for d in unassigned:
                union |= d["mask_binary"].astype(bool)
            panels.append({
                "panel_key": manual_panel,
                "part_class": "(dipilih manual)",
                "part_mask": None,
                "damages": unassigned,
                "mask_area_ratio": float(union.sum() / union.size),
            })
            unassigned = []

        items = []
        for p in panels:
            d_depth = self._panel_depth(image, p)
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
        }

    def _panel_depth(self, image, panel):
        # kedalaman hanya untuk kelas di kelas_kedalaman (Dent), diambil penyok terdalam
        depth_dmgs = [d for d in panel["damages"] if d["class_name"] in self.calculator.depth_classes]

        d_depth = 0.0
        for d in depth_dmgs:
            mask = d["mask_binary"].astype(bool)
            if panel["part_mask"] is not None:
                mask &= panel["part_mask"]
            if mask.sum() < 50:
                continue
            d_depth = max(d_depth, self.depth_estimator.crop_depth_deviation(image, mask, DEPTH_CROP_MARGIN))
        return d_depth

    def analyze(self, image_bgr, manual_panel=None, damage_conf=None, parts_conf=None):
        return self.estimate(self.run_models(image_bgr, damage_conf, parts_conf), manual_panel)

    def draw_overlay(self, image, part_dets, damage_dets):
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
