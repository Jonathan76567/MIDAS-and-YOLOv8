"""
Pengaruh ambang confidence model kerusakan terhadap precision, recall, dan deteksi palsu
(split test dataset kerusakan, dipakai untuk pengujian faktor deteksi di Bab 4).

- Prediksi dibuat sekali dengan ambang rendah (0,05), lalu disaring untuk setiap ambang.
- Prediksi benar (TP) bila kelasnya sama dan IoU mask dengan anotasi >= 0,5
  (dipasangkan dari confidence tertinggi, satu anotasi hanya boleh dipakai sekali).
- Foto tanpa anotasi Dent/Scratch = foto negatif. FPR = foto negatif yang memunculkan
  minimal satu deteksi / seluruh foto negatif.

Jalankan dari root project:
    python -m src.evaluate_threshold
"""
import json
import os

import cv2
import numpy as np

from src.convert_dataset import OUT_DIR
from src.detector import SegDetector

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
IOU_MIN = 0.5


def load_gt(label_path, h, w, names):
    """Label YOLO-seg (poligon ternormalisasi) -> list (nama kelas, mask bool)."""
    gts = []
    if not os.path.exists(label_path):
        return gts
    for line in open(label_path).read().splitlines():
        vals = line.split()
        if len(vals) < 7:
            continue
        pts = (np.array(vals[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        if mask.any():
            gts.append((names[int(vals[0])], mask.astype(bool)))
    return gts


def match(preds, gts):
    """preds: list (kelas, conf, mask) urut conf menurun. Return (tp, fp, fn)."""
    used = [False] * len(gts)
    tp = 0
    for cls, _, pm in preds:
        best, best_iou = None, IOU_MIN
        for j, (gcls, gm) in enumerate(gts):
            if used[j] or gcls != cls:
                continue
            inter = np.logical_and(pm, gm).sum()
            if inter == 0:
                continue
            iou = inter / np.logical_or(pm, gm).sum()
            if iou >= best_iou:
                best, best_iou = j, iou
        if best is not None:
            used[best] = True
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def evaluate(weights=os.path.join('weights', 'damage_best.pt')):
    root = os.path.join(OUT_DIR, 'damage')
    names = ['Dent', 'Scratch']
    det = SegDetector(weights, conf_threshold=min(THRESHOLDS))
    img_dir = os.path.join(root, 'images', 'test')

    stats = {src: {t: {'tp': 0, 'fp': 0, 'fn': 0, 'neg': 0, 'neg_fp': 0} for t in THRESHOLDS}
             for src in ('semua', 'kaggle', 'cardd')}
    for k, f in enumerate(sorted(os.listdir(img_dir)), 1):
        img = cv2.imread(os.path.join(img_dir, f))
        h, w = img.shape[:2]
        gts = load_gt(os.path.join(root, 'labels', 'test', os.path.splitext(f)[0] + '.txt'), h, w, names)
        preds = sorted(((d['class_name'], d['confidence'], d['mask_binary'].astype(bool))
                        for d in det.predict(img)), key=lambda x: -x[1])
        src = 'cardd' if f.startswith('cardd_') else 'kaggle'
        for t in THRESHOLDS:
            p_t = [p for p in preds if p[1] >= t]
            tp, fp, fn = match(p_t, gts)
            for s in ('semua', src):
                st = stats[s][t]
                st['tp'] += tp
                st['fp'] += fp
                st['fn'] += fn
                if not gts:
                    st['neg'] += 1
                    st['neg_fp'] += bool(p_t)
        if k % 50 == 0:
            print(f'[INFO] {k} foto diproses')

    summary = {}
    for s, by_t in stats.items():
        summary[s] = {}
        for t, st in by_t.items():
            p = st['tp'] / (st['tp'] + st['fp']) if st['tp'] + st['fp'] else 0.0
            r = st['tp'] / (st['tp'] + st['fn']) if st['tp'] + st['fn'] else 0.0
            summary[s][t] = {
                **st,
                'precision': round(p, 4),
                'recall': round(r, 4),
                'f1': round(2 * p * r / (p + r), 4) if p + r else 0.0,
                'fpr_negatif': round(st['neg_fp'] / st['neg'], 4) if st['neg'] else None,
            }
    return summary


if __name__ == '__main__':
    result = evaluate()
    for s, by_t in result.items():
        print(f'\n=== {s} ===')
        print('ambang  precision  recall  f1     FPR foto negatif')
        for t, r in by_t.items():
            print(f'{t:<7} {r["precision"]:<10} {r["recall"]:<7} {r["f1"]:<6} {r["fpr_negatif"]} '
                  f'({r["neg_fp"]}/{r["neg"]})')
    with open(os.path.join('runs', 'uji_ambang.json'), 'w') as f:
        json.dump(result, f, indent=2)
