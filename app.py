"""
Web app estimasi biaya body repair.
Jalankan: streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import BodyRepairPipeline

st.set_page_config(page_title="Estimasi Biaya Body Repair", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Memuat model (YOLOv8-Seg & MiDaS)...")
def load_pipeline():
    return BodyRepairPipeline()


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


pipeline = load_pipeline()
calc = pipeline.calculator

# ---------------- Sidebar ----------------
st.sidebar.header("Pengaturan")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.markdown("**Formula biaya**")
st.sidebar.latex(r"P = P_{base}\,(1 + \alpha A_{mask} + \beta D_{depth})")
st.sidebar.caption(f"α = {calc.alpha}, β = {calc.beta}")
with st.sidebar.expander("Harga dasar panel"):
    st.table(pd.DataFrame(
        [(k, f"Rp {v:,.0f}".replace(",", ".")) for k, v in calc.db["harga_base_panel"].items()],
        columns=["Panel", "Harga dasar"],
    ))

# ---------------- Main ----------------
st.title("🚗 Estimasi Biaya Body Repair")
st.caption("Upload foto kerusakan mobil. Sistem mendeteksi kerusakan (YOLOv8-Seg), "
           "mengenali panel, mengukur kedalaman penyok (MiDaS), lalu menghitung estimasi biaya.")

if pipeline.using_placeholder:
    st.warning("Model kerusakan belum dilatih. Saat ini memakai **yolov8s-seg.pt (COCO)** sebagai "
               "placeholder, sehingga yang terdeteksi adalah objek umum, bukan kerusakan.")
if pipeline.parts_detector is None:
    st.info("Model panel (`weights/parts_best.pt`) belum ada, panel dipilih manual.")

uploaded = st.file_uploader("Foto mobil", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

# Inferensi model hanya diulang jika foto atau threshold berubah
cache_key = (uploaded.file_id, conf)
if st.session_state.get("cache_key") != cache_key:
    image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    with st.spinner("Menganalisis foto..."):
        st.session_state.raw = pipeline.run_models(image, conf=conf)
    st.session_state.cache_key = cache_key
raw = st.session_state.raw

# Kerusakan tanpa panel -> minta pengguna memilih panel
preview = pipeline.estimate(raw)
manual_panel = None
if preview["unassigned"]:
    names = ", ".join(sorted({d["class_name"] for d in preview["unassigned"]}))
    manual_panel = st.selectbox(
        f"{len(preview['unassigned'])} kerusakan ({names}) tidak menempel ke panel yang terdeteksi. "
        "Pilih panelnya:",
        calc.panel_names(),
    )
result = pipeline.estimate(raw, manual_panel) if manual_panel else preview

col1, col2, col3 = st.columns(3)
col1.image(to_rgb(raw["image"]), caption="Foto asli", width="stretch")
col2.image(to_rgb(result["overlay"]), caption="Deteksi (merah = kerusakan, biru = panel)",
           width="stretch")
col3.image(to_rgb(result["depth_vis"]), caption="Depth map MiDaS", width="stretch")

st.subheader("Rincian estimasi")
if not result["items"]:
    st.info("Tidak ada kerusakan yang terdeteksi.")
else:
    rows = []
    for item in result["items"]:
        c = item["cost"]
        rows.append({
            "Panel": item["panel_key"] or "(harga belum ada)",
            "Kelas panel": item["part_class"],
            "Kerusakan": ", ".join(item["damage_classes"]),
            "Luas A_mask": f"{item['mask_area_ratio']*100:.1f}%",
            "Kedalaman D_depth": round(item["depth_deviation"], 3),
            "Pengganda": f"{c['parameters']['multiplier']}x" if c else "-",
            "Harga dasar": c["formatted_base"] if c else "-",
            "Estimasi": c["formatted_total"] if c else item["status"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.metric("Total estimasi biaya", result["formatted_total"])
