import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import BodyRepairPipeline

st.set_page_config(page_title="Estimasi Biaya Body Repair", page_icon="🚗", layout="wide")

NAMA_KERUSAKAN = {"Dent": "Penyok", "Scratch": "Baret"}
PERINGATAN = ("Harga di atas hanya estimasi awal dan belum menjadi harga final. "
              "Untuk pengecekan dan harga pasti, silakan bawa kendaraan ke bengkel.")


@st.cache_resource(show_spinner="Memuat model...")
def load_pipeline():
    return BodyRepairPipeline()


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def nama_panel(key):
    return key.replace("_", " ").title()


def rupiah(nilai):
    return f"Rp {nilai:,.0f}".replace(",", ".")


pipeline = load_pipeline()
calc = pipeline.calculator

st.sidebar.header("Informasi")
st.sidebar.subheader("Harga dasar per panel")
st.sidebar.dataframe(
    pd.DataFrame(
        [(nama_panel(k), rupiah(v)) for k, v in calc.db["harga_base_panel"].items()],
        columns=["Panel", "Harga dasar"],
    ),
    hide_index=True,
    width="stretch",
)
st.sidebar.warning(PERINGATAN, icon="⚠️")

st.title("🚗 Estimasi Biaya Body Repair")
st.caption("Upload foto kerusakan mobil. Sistem akan mendeteksi penyok dan baret, mengenali panel "
           "yang rusak, lalu menghitung estimasi biaya perbaikan.")

uploaded = st.file_uploader("Foto mobil", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

cache_key = uploaded.file_id
if st.session_state.get("cache_key") != cache_key:
    image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    with st.spinner("Menganalisis foto..."):
        st.session_state.raw = pipeline.run_models(image)
    st.session_state.cache_key = cache_key
raw = st.session_state.raw

preview = pipeline.estimate(raw)
manual_panel = None
if preview["unassigned"]:
    manual_panel = st.selectbox(
        "Sebagian kerusakan tidak dapat dikenali panelnya. Pilih panel yang rusak:",
        calc.panel_names(),
        format_func=nama_panel,
    )
result = pipeline.estimate(raw, manual_panel) if manual_panel else preview

col1, col2 = st.columns(2)
col1.image(to_rgb(raw["image"]), caption="Foto asli", width="stretch")
col2.image(to_rgb(result["overlay"]), caption="Hasil deteksi (merah = kerusakan, biru = panel)",
           width="stretch")

st.subheader("Rincian estimasi")
if not result["items"]:
    st.info("Tidak ada kerusakan yang terdeteksi.")
else:
    rows = []
    for item in result["items"]:
        c = item["cost"]
        rows.append({
            "Panel": nama_panel(item["panel_key"]) if item["panel_key"] else item["part_class"],
            "Kerusakan": ", ".join(NAMA_KERUSAKAN.get(d, d) for d in item["damage_classes"]),
            "Estimasi biaya": c["formatted_total"] if c else "Perlu pengecekan di bengkel",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.metric("Total estimasi biaya", result["formatted_total"])
st.warning(PERINGATAN, icon="⚠️")
