import numpy as np


def match(damage_dets, part_dets, panel_mapping):
    """
    Memasangkan setiap kerusakan ke panel body dengan overlap piksel terbesar,
    lalu mengelompokkan kerusakan per panel (satu panel = satu item biaya).

    Param:
    - damage_dets: list deteksi dari model kerusakan
    - part_dets: list deteksi dari model panel
    - panel_mapping: dict kelas model panel -> key harga di price_list
      (mis. {"Front-door": "pintu_depan", "Roof": None}). Nilai None berarti panel
      body yang harganya belum tersedia. Kelas yang tidak ada di mapping (kaca,
      lampu, roda, grille, dll.) diabaikan karena bukan pekerjaan body repair.

    Returns:
    - panels: list dict {panel_key, part_class, part_mask, damage_mask, damages, mask_area_ratio}
      (panel_key bisa None bila harga panel belum tersedia)
    - unassigned: list deteksi kerusakan yang tidak menempel ke panel body manapun
    """
    body_parts = [p for p in part_dets if p["class_name"] in panel_mapping]

    groups = {}  # index part -> list kerusakan
    unassigned = []

    for dmg in damage_dets:
        dmg_mask = dmg["mask_binary"].astype(bool)
        best_idx, best_overlap = None, 0

        for idx, part in enumerate(body_parts):
            overlap = int(np.logical_and(dmg_mask, part["mask_binary"].astype(bool)).sum())
            if overlap > best_overlap:
                best_idx, best_overlap = idx, overlap

        if best_idx is None:
            unassigned.append(dmg)
        else:
            groups.setdefault(best_idx, []).append(dmg)

    panels = []
    for idx, dmgs in groups.items():
        part = body_parts[idx]
        part_mask = part["mask_binary"].astype(bool)

        # Gabungan semua mask kerusakan pada panel ini, dipotong ke area panel
        damage_union = np.zeros_like(part_mask)
        for d in dmgs:
            damage_union |= d["mask_binary"].astype(bool)
        damage_in_panel = damage_union & part_mask

        panels.append({
            "panel_key": panel_mapping[part["class_name"]],
            "part_class": part["class_name"],
            "part_mask": part_mask,
            "damage_mask": damage_in_panel,
            "damages": dmgs,
            # A_mask = |kerusakan ∩ panel| / |panel|
            "mask_area_ratio": float(damage_in_panel.sum() / part_mask.sum()),
        })

    return panels, unassigned
