"""
CLI estimasi biaya body repair.

Contoh:
    python main.py foto.jpg
    python main.py foto.jpg --panel pintu_depan --conf 0.3
"""
import argparse
import os

import cv2

from src.pipeline import BodyRepairPipeline


def main():
    parser = argparse.ArgumentParser(description="Estimasi biaya body repair dari foto kerusakan mobil")
    parser.add_argument("image", help="path foto mobil")
    parser.add_argument("--panel", help="panel manual untuk kerusakan yang tidak menempel ke panel terdeteksi")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold YOLO")
    parser.add_argument("--out", default="output", help="folder untuk menyimpan gambar hasil")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {args.image}")

    pipeline = BodyRepairPipeline(conf_threshold=args.conf)
    if pipeline.using_placeholder:
        print("[PERINGATAN] Model kerusakan belum dilatih, memakai yolov8s-seg.pt (COCO) sebagai placeholder.")
    if pipeline.parts_detector is None:
        print("[PERINGATAN] Model panel belum ada (weights/parts_best.pt), panel harus dipilih dengan --panel.")

    result = pipeline.analyze(image, manual_panel=args.panel)

    print("\n=== HASIL ESTIMASI BIAYA ===")
    if not result["items"]:
        print("Tidak ada kerusakan yang bisa dihitung.")
    for i, item in enumerate(result["items"], 1):
        print(f"\n#{i} Panel {item['panel_key'] or '(harga belum ada)'} ({item['part_class']})")
        print(f"   Kerusakan      : {', '.join(item['damage_classes'])} ({item['n_damages']} area)")
        print(f"   Luas (A_mask)  : {item['mask_area_ratio']:.3f}")
        print(f"   Kedalaman (D)  : {item['depth_deviation']:.3f}")
        if item["cost"]:
            c = item["cost"]
            print(f"   Harga dasar    : {c['formatted_base']}")
            print(f"   Pengganda      : {c['parameters']['multiplier']}x")
            print(f"   Estimasi biaya : {c['formatted_total']}")
        else:
            print(f"   Error          : {item['status']}")

    if result["unassigned"]:
        names = ", ".join(d["class_name"] for d in result["unassigned"])
        print(f"\n[INFO] {len(result['unassigned'])} kerusakan belum punya panel ({names}). "
              f"Gunakan --panel <nama> dengan pilihan: {', '.join(pipeline.calculator.panel_names())}")

    print(f"\nTOTAL ESTIMASI: {result['formatted_total']}")

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{base}_overlay.jpg"), result["overlay"])
    cv2.imwrite(os.path.join(args.out, f"{base}_depth.jpg"), result["depth_vis"])
    print(f"[INFO] Gambar hasil disimpan di folder '{args.out}/'")


if __name__ == "__main__":
    main()
