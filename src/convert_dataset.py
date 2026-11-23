import argparse
import base64
import io
import json
import os
import random
import re
import shutil
import zlib

import cv2
import numpy as np
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
OUT_DIR = os.path.join(ROOT_DIR, 'data', 'annotated')

# nama folder di dataset asli tertukar
DATASETS = {
    'parts': 'Car damages dataset',
    'damage': 'Car parts dataset',
}
SPLITS = ('train', 'val', 'test')

CLASS_FILTER = {
    'parts': None,
    'damage': ['Dent', 'Scratch'],
}

CARDD_DIR = os.path.join(RAW_DIR, 'CarDD_hf')
CARDD_LABEL_MAP = {'dent': 'Dent', 'scratch': 'Scratch'}


def safe_stem(filename):
    stem = os.path.splitext(filename)[0]
    return re.sub(r'[^0-9a-zA-Z]+', '_', stem).strip('_').lower()


def split_names(names, ratios, rng):
    names = sorted(names)
    rng.shuffle(names)
    n_train = round(len(names) * ratios[0])
    n_val = round(len(names) * ratios[1])
    return {
        'train': names[:n_train],
        'val': names[n_train:n_train + n_val],
        'test': names[n_train + n_val:],
    }


def make_split(image_sets, ratios, seed):
    rng = random.Random(seed)
    parts, damage = image_sets['parts'], image_sets['damage']
    groups = {'shared': parts & damage, 'parts_only': parts - damage, 'damage_only': damage - parts}

    assignment = {}
    for group in groups.values():
        for split, names in split_names(group, ratios, rng).items():
            for n in names:
                assignment[n] = split
    return assignment, groups['shared']


def ann_to_yolo_lines(ann, class_index):
    w, h = ann['size']['width'], ann['size']['height']
    lines = []
    for obj in ann['objects']:
        if obj['geometryType'] != 'polygon' or obj['classTitle'] not in class_index:
            continue
        pts = obj['points']['exterior']
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.append(f'{min(max(x / w, 0.0), 1.0):.6f}')
            coords.append(f'{min(max(y / h, 0.0), 1.0):.6f}')
        lines.append(f'{class_index[obj["classTitle"]]} ' + ' '.join(coords))
    return lines


def load_cardd_samples():
    path = os.path.join(CARDD_DIR, 'samples.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)['samples']


def cardd_split(samples, ratios, seed):
    names = [os.path.basename(s['filepath']) for s in samples]
    assignment = {}
    for split, group in split_names(names, ratios, random.Random(seed + 1)).items():
        for n in group:
            assignment[n] = split
    return assignment


def cardd_mask_polygons(det, img_w, img_h):
    # mask FiftyOne: array seukuran bounding box, dikompres zlib
    raw = zlib.decompress(base64.b64decode(det['mask']['$binary']['base64']))
    mask = np.load(io.BytesIO(raw)).astype(np.uint8)

    bx, by = det['bounding_box'][:2]
    x0, y0 = int(round(bx * img_w)), int(round(by * img_h))
    h, w = min(mask.shape[0], img_h - y0), min(mask.shape[1], img_w - x0)
    full = np.zeros((img_h, img_w), np.uint8)
    full[y0:y0 + h, x0:x0 + w] = mask[:h, :w]

    polygons = []
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            polygons.append(c)
    return polygons


def cardd_to_yolo_lines(sample, class_index):
    w, h = sample['metadata']['width'], sample['metadata']['height']
    lines = []
    for det in (sample.get('segmentations') or {}).get('detections', []):
        name = CARDD_LABEL_MAP.get(det['label'])
        if name not in class_index or not det.get('mask'):
            continue
        for poly in cardd_mask_polygons(det, w, h):
            coords = ' '.join(f'{min(max(x / w, 0.0), 1.0):.6f} {min(max(y / h, 0.0), 1.0):.6f}' for x, y in poly)
            lines.append(f'{class_index[name]} {coords}')
    return lines


def write_source_lists(dst, class_names, files_by_source):
    for source, by_split in files_by_source.items():
        for split, files in by_split.items():
            with open(os.path.join(dst, f'{split}_{source}.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(f'./images/{split}/{n}' for n in sorted(files)))
        data_yaml = {
            'path': dst.replace('\\', '/'),
            'train': 'images/train',
            'val': f'val_{source}.txt',
            'test': f'test_{source}.txt',
            'names': dict(enumerate(class_names)),
        }
        with open(os.path.join(dst, f'data_{source}.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)


def convert(ratios=(0.7, 0.15, 0.15), seed=42, only=None, use_cardd=True):
    cardd_samples = load_cardd_samples() if use_cardd else None
    cardd_assignment = cardd_split(cardd_samples, ratios, seed) if cardd_samples else {}

    if cardd_samples and (not only or 'damage' in only):
        missing = [s['filepath'] for s in cardd_samples
                   if not os.path.exists(os.path.join(CARDD_DIR, s['filepath']))]
        if missing:
            raise FileNotFoundError(f'{len(missing)} gambar CarDD belum diunduh (mis. {missing[0]}). '
                                    'Unduh ulang, atau jalankan dengan --no-cardd.')
    image_sets, sources = {}, {}
    for key, folder in DATASETS.items():
        src = os.path.join(RAW_DIR, folder, 'File1')
        if not os.path.isdir(src):
            raise FileNotFoundError(f'Folder dataset tidak ditemukan: {src}')
        sources[key] = src
        image_sets[key] = set(os.listdir(os.path.join(src, 'img')))

    assignment, shared = make_split(image_sets, ratios, seed)

    summary = {}
    for key, folder in DATASETS.items():
        if only and key not in only:
            continue
        src = sources[key]
        meta_path = os.path.join(RAW_DIR, folder, 'meta.json')
        with open(meta_path, encoding='utf-8') as f:
            class_names = [c['title'] for c in json.load(f)['classes']]
        if CLASS_FILTER.get(key):
            missing = set(CLASS_FILTER[key]) - set(class_names)
            if missing:
                raise ValueError(f'Kelas {missing} tidak ada di {meta_path}')
            class_names = list(CLASS_FILTER[key])
        class_index = {name: i for i, name in enumerate(class_names)}

        dst = os.path.join(OUT_DIR, key)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        for split in SPLITS:
            os.makedirs(os.path.join(dst, 'images', split))
            os.makedirs(os.path.join(dst, 'labels', split))

        seen_stems = {}
        counts = {s: 0 for s in SPLITS}
        files_by_source = {'kaggle': {s: [] for s in SPLITS}}
        for img_name in sorted(image_sets[key]):
            stem = safe_stem(img_name)
            if stem in seen_stems:
                raise ValueError(f'Nama file bentrok: {img_name} dan {seen_stems[stem]}')
            seen_stems[stem] = img_name

            with open(os.path.join(src, 'ann', img_name + '.json'), encoding='utf-8') as f:
                ann = json.load(f)
            split = assignment[img_name]
            ext = os.path.splitext(img_name)[1].lower()

            shutil.copy(os.path.join(src, 'img', img_name), os.path.join(dst, 'images', split, stem + ext))
            with open(os.path.join(dst, 'labels', split, stem + '.txt'), 'w') as f:
                f.write('\n'.join(ann_to_yolo_lines(ann, class_index)))
            counts[split] += 1
            files_by_source['kaggle'][split].append(stem + ext)

        if key == 'damage' and cardd_samples:
            files_by_source['cardd'] = {s: [] for s in SPLITS}
            for sample in cardd_samples:
                img_name = os.path.basename(sample['filepath'])
                img_path = os.path.join(CARDD_DIR, sample['filepath'])
                split = cardd_assignment[img_name]
                out_name = 'cardd_' + img_name

                shutil.copy(img_path, os.path.join(dst, 'images', split, out_name))
                with open(os.path.join(dst, 'labels', split, os.path.splitext(out_name)[0] + '.txt'), 'w') as f:
                    f.write('\n'.join(cardd_to_yolo_lines(sample, class_index)))
                counts[split] += 1
                files_by_source['cardd'][split].append(out_name)

        if key == 'damage':
            write_source_lists(dst, class_names, files_by_source)

        data_yaml = {
            'path': dst.replace('\\', '/'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': dict(enumerate(class_names)),
        }
        with open(os.path.join(dst, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

        summary[key] = {'classes': class_names, 'images': counts}

    shared_split = {s: sorted(n for n in shared if assignment[n] == s) for s in SPLITS}
    with open(os.path.join(OUT_DIR, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'seed': seed,
            'ratios': ratios,
            'raw_folders': DATASETS,
            'shared_images': shared_split,
            'cardd_images': {s: sorted(n for n, sp in cardd_assignment.items() if sp == s) for s in SPLITS},
        }, f, indent=2)

    summary['shared'] = {s: len(v) for s, v in shared_split.items()}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Konversi dataset Supervisely -> YOLO-seg')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', choices=list(DATASETS), help='konversi ulang satu dataset saja')
    parser.add_argument('--no-cardd', action='store_true', help='jangan tambahkan CarDD ke dataset kerusakan')
    args = parser.parse_args()

    result = convert(seed=args.seed, only=[args.only] if args.only else None, use_cardd=not args.no_cardd)
    for key in DATASETS:
        if key not in result:
            continue
        print(f"[{key}] {len(result[key]['classes'])} kelas, gambar per split: {result[key]['images']}")
        print(f"        kelas: {result[key]['classes']}")
    print(f"[shared] foto dengan label panel & kerusakan per split: {result['shared']}")
    print(f'[INFO] Hasil tersimpan di {OUT_DIR}')
