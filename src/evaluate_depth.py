"""
Uji pengukuran kedalaman: apakah D_depth penyok (Dent) lebih besar daripada baret (Scratch)?
Dipakai untuk pengujian pengukuran kedalaman di Bab 4.

D_depth dihitung pada mask anotasi split test dataset kerusakan dengan dua cara:
- sederhana : depth map seluruh foto, |rata-rata mask - rata-rata cincin| / (maks - min depth foto)
- sistem    : DepthEstimator.crop_depth_deviation (crop + permukaan kuadratik), sama dengan pipeline
Kemampuan memisahkan Dent dan Scratch diukur dengan AUC (peluang D_depth sebuah penyok lebih besar
daripada D_depth sebuah baret; 0,5 = tebakan acak, 1 = sempurna) dan rentang kepercayaan 95% bootstrap.

Jalankan dari root project:
    python -m src.evaluate_depth
"""
import json
import os

import cv2
import numpy as np

from src.convert_dataset import OUT_DIR
from src.depth_estimator import DepthEstimator
from src.evaluate_threshold import load_gt

MIN_PIXELS = 50  # sama dengan pipeline: kerusakan lebih kecil dari ini tidak diukur


def simple_deviation(depth_map, mask):
    k = max(7, int(0.15 * np.sqrt(mask.sum())))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
    span = depth_map.max() - depth_map.min()
    if ring.sum() < 10 or span <= 1e-6:
        return 0.0
    return float(abs(depth_map[mask].mean() - depth_map[ring].mean()) / span)


def auc(pos, neg):
    """AUC Mann-Whitney: peluang nilai pos > nilai neg (seri dihitung 0,5)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # rata-rata ranking untuk nilai seri
        idx = allv == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    vals = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def evaluate():
    root = os.path.join(OUT_DIR, 'damage')
    img_dir = os.path.join(root, 'images', 'test')
    est = DepthEstimator()
    values = {'sederhana': {'Dent': [], 'Scratch': []}, 'sistem': {'Dent': [], 'Scratch': []}}

    for k, f in enumerate(sorted(os.listdir(img_dir)), 1):
        img = cv2.imread(os.path.join(img_dir, f))
        h, w = img.shape[:2]
        gts = [(c, m) for c, m in load_gt(os.path.join(root, 'labels', 'test', os.path.splitext(f)[0] + '.txt'),
                                           h, w, ['Dent', 'Scratch']) if m.sum() >= MIN_PIXELS]
        if not gts:
            continue
        full = est.estimate(img)
        for cls, m in gts:
            values['sederhana'][cls].append(simple_deviation(full, m))
            values['sistem'][cls].append(est.crop_depth_deviation(img, m))
        if k % 50 == 0:
            print(f'[INFO] {k} foto diproses')

    summary = {}
    for method, v in values.items():
        dent = np.array(v['Dent'])
        summary[method] = {
            'n_dent': len(v['Dent']),
            'n_scratch': len(v['Scratch']),
            'auc': round(auc(v['Dent'], v['Scratch']), 3),
            'auc_ci95': bootstrap_ci(v['Dent'], v['Scratch']),
            'dent_median': round(float(np.median(dent)), 3),
            'dent_p75': round(float(np.percentile(dent, 75)), 3),
            'dent_p95': round(float(np.percentile(dent, 95)), 3),
            'scratch_median': round(float(np.median(v['Scratch'])), 3),
        }
    return summary


if __name__ == '__main__':
    result = evaluate()
    for method, r in result.items():
        print(f'{method:10s}: {r}')
    with open(os.path.join('runs', 'uji_kedalaman.json'), 'w') as f:
        json.dump(result, f, indent=2)
