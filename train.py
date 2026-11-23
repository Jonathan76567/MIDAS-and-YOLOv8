import argparse
import os
import shutil

import torch
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT_DIR, 'runs')
DATA = {
    'parts': os.path.join(ROOT_DIR, 'data', 'annotated', 'parts', 'data.yaml'),
    'damage': os.path.join(ROOT_DIR, 'data', 'annotated', 'damage', 'data.yaml'),
}


def train_one(name, args):
    last_ckpt = os.path.join(RUNS_DIR, name, 'weights', 'last.pt')
    dst = os.path.join(ROOT_DIR, 'weights', f'{name}_best.pt')

    # epoch -1 = training sudah selesai
    if args.resume and os.path.exists(last_ckpt):
        if torch.load(last_ckpt, map_location='cpu', weights_only=False).get('epoch') == -1:
            print(f'[INFO] Training {name} sudah selesai sebelumnya, dilewati')
            return

    if args.resume and os.path.exists(last_ckpt):
        print(f'[INFO] Melanjutkan training {name} dari {last_ckpt}')
        results = YOLO(last_ckpt).train(resume=True)
    else:
        if args.resume:
            print(f'[INFO] Checkpoint {name} tidak ditemukan, mulai dari awal')
        elif os.path.exists(last_ckpt):
            os.remove(last_ckpt)
        print(f'[INFO] Training {name} dari bobot awal {args.weights}')
        results = YOLO(args.weights).train(
            data=DATA[name],
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            fraction=args.fraction,
            patience=args.patience,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=args.workers,
            project=RUNS_DIR,
            name=name,
            exist_ok=True,
        )

    best = os.path.join(results.save_dir, 'weights', 'best.pt')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(best, dst)
    print(f'[INFO] Model {name} tersimpan: {dst}')


def main():
    parser = argparse.ArgumentParser(description='Training YOLOv8-Seg panel & kerusakan')
    parser.add_argument('--model', choices=['parts', 'damage', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--fraction', type=float, default=1.0, help='porsi data train (untuk uji coba)')
    parser.add_argument('--patience', type=int, default=15, help='berhenti jika val tidak membaik N epoch')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='lanjutkan dari runs/<model>/weights/last.pt')
    parser.add_argument('--weights', default=os.path.join(ROOT_DIR, 'yolov8s-seg.pt'),
                        help='bobot awal (default pretrained COCO)')
    args = parser.parse_args()

    for name, path in DATA.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} belum ada. Jalankan dulu: python -m src.convert_dataset')

    names = ['parts', 'damage'] if args.model == 'all' else [args.model]
    for name in names:
        train_one(name, args)


if __name__ == '__main__':
    main()
