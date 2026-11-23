import numpy as np


def match(damage_dets, part_dets, panel_mapping):
    body_parts = [p for p in part_dets if p["class_name"] in panel_mapping]

    groups = {}
    unassigned = []

    # tiap kerusakan masuk ke panel dengan overlap terbesar
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

        damage_union = np.zeros_like(part_mask)
        for d in dmgs:
            damage_union |= d["mask_binary"].astype(bool)
        damage_in_panel = damage_union & part_mask

        panels.append({
            "panel_key": panel_mapping[part["class_name"]],
            "part_class": part["class_name"],
            "part_mask": part_mask,
            "damages": dmgs,
            "mask_area_ratio": float(damage_in_panel.sum() / part_mask.sum()),
        })

    return panels, unassigned
