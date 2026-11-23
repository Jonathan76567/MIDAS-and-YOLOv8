import argparse
import os

import cv2

from src.pipeline import DAMAGE_CONF, PARTS_CONF, BodyRepairPipeline


def main():
    parser = argparse.ArgumentParser(description="Estimasi biaya body repair dari foto kerusakan mobil")
    parser.add_argument("image", help="path foto mobil")
    parser.add_argument("--panel", help="panel manual untuk kerusakan yang tidak menempel ke panel terdeteksi")
    parser.add_argument("--damage-conf", type=float, default=DAMAGE_CONF, help="confidence threshold model kerusakan")
    parser.add_argument("--parts-conf", type=float, default=PARTS_CONF, help="confidence threshold model panel")
    parser.add_argument("--out", default="output", help="folder untuk menyimpan gambar hasil")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {args.image}")

    pipeline = BodyRepairPipeline(damage_conf=args.damage_conf, parts_conf=args.parts_conf)
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
    print(f"[INFO] Gambar hasil disimpan di folder '{args.out}/'")


if __name__ == "__main__":
    main()
