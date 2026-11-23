import json
import os

import cv2
import numpy as np
import pandas as pd

from src.convert_dataset import CLASS_FILTER, DATASETS, OUT_DIR, RAW_DIR


def load_gt_dets(ann_path, classes=None):
    with open(ann_path, encoding='utf-8') as f:
        ann = json.load(f)
    h, w = ann['size']['height'], ann['size']['width']

    dets = []
    for obj in ann['objects']:
        if obj['geometryType'] != 'polygon' or len(obj['points']['exterior']) < 3:
            continue
        if classes is not None and obj['classTitle'] not in classes:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.array(obj['points']['exterior'], np.int32)], 1)
        for hole in obj['points']['interior']:
            cv2.fillPoly(mask, [np.array(hole, np.int32)], 0)
        if mask.sum() == 0:
            continue
        ys, xs = np.nonzero(mask)
        dets.append({
            'class_name': obj['classTitle'],
            'confidence': 1.0,
            'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            'mask_binary': mask,
        })
    return dets


def _costs_by_panel(result):
    out = {}
    for item in result['items']:
        if not item['cost']:
            continue
        key = item['panel_key']
        prev = out.get(key, {'cost': 0, 'a_mask': 0.0, 'd_depth': 0.0})
        out[key] = {
            'cost': prev['cost'] + item['cost']['total_estimated_cost'],
            'a_mask': max(prev['a_mask'], item['mask_area_ratio']),
            'd_depth': max(prev['d_depth'], item['depth_deviation']),
        }
    return out


def evaluate_end_to_end(pipeline, split='test', limit=None):
    # foto yang punya label panel dan kerusakan sekaligus
    with open(os.path.join(OUT_DIR, 'split_info.json'), encoding='utf-8') as f:
        names = json.load(f)['shared_images'][split]
    if limit:
        names = names[:limit]

    parts_dir = os.path.join(RAW_DIR, DATASETS['parts'], 'File1')
    damage_dir = os.path.join(RAW_DIR, DATASETS['damage'], 'File1')

    rows = []
    for name in names:
        image = cv2.imread(os.path.join(parts_dir, 'img', name))
        pred_raw = pipeline.run_models(image)
        gt_raw = {
            'image': image,
            'damage_dets': load_gt_dets(os.path.join(damage_dir, 'ann', name + '.json'), CLASS_FILTER['damage']),
            'part_dets': load_gt_dets(os.path.join(parts_dir, 'ann', name + '.json'), CLASS_FILTER['parts']),
        }

        gt = _costs_by_panel(pipeline.estimate(gt_raw))
        pred = _costs_by_panel(pipeline.estimate(pred_raw))

        for key in sorted(set(gt) | set(pred)):
            g, p = gt.get(key), pred.get(key)
            rows.append({
                'image': name,
                'panel': key,
                'in_gt': g is not None,
                'in_pred': p is not None,
                'gt_a_mask': g['a_mask'] if g else 0.0,
                'pred_a_mask': p['a_mask'] if p else 0.0,
                'gt_d_depth': g['d_depth'] if g else 0.0,
                'pred_d_depth': p['d_depth'] if p else 0.0,
                'gt_cost': g['cost'] if g else 0,
                'pred_cost': p['cost'] if p else 0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    tp = int((df.in_gt & df.in_pred).sum())
    fp = int((~df.in_gt & df.in_pred).sum())
    fn = int((df.in_gt & ~df.in_pred).sum())
    both = df[df.in_gt & df.in_pred]
    per_image = df.groupby('image')[['gt_cost', 'pred_cost']].sum()
    per_image_gt = per_image[per_image.gt_cost > 0]

    summary = {
        'n_images': len(names),
        'panel_precision': round(tp / (tp + fp), 4) if tp + fp else 0.0,
        'panel_recall': round(tp / (tp + fn), 4) if tp + fn else 0.0,
        'a_mask_mae (panel cocok)': round(float((both.gt_a_mask - both.pred_a_mask).abs().mean()), 4) if len(both) else None,
        'cost_mae_per_panel (Rp)': round(float((df.gt_cost - df.pred_cost).abs().mean())),
        'total_cost_mae_per_image (Rp)': round(float((per_image.gt_cost - per_image.pred_cost).abs().mean())),
        'total_cost_mape_per_image (%)': round(float(
            ((per_image_gt.gt_cost - per_image_gt.pred_cost).abs() / per_image_gt.gt_cost).mean() * 100), 2)
            if len(per_image_gt) else None,
    }
    return df, summary


if __name__ == '__main__':
    from src.pipeline import BodyRepairPipeline

    pipe = BodyRepairPipeline()
    per_panel, summary = evaluate_end_to_end(pipe)
    print(per_panel.to_string(index=False))
    print('\n=== RINGKASAN ===')
    for k, v in summary.items():
        print(f'{k:32s}: {v}')
