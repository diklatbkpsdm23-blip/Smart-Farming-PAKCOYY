# =========================================================
# SMART FARMING PAKCOY 🥬 — Streamlit Dashboard
# Monitoring & Klasifikasi AI untuk Budidaya Pakcoy
# Disesuaikan dengan Dataset_Pakcoy.xlsx
# =========================================================

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# =========================================================
# KONFIGURASI MODEL CNN 2D (DETEKSI FOTO DAUN)
# =========================================================
# File .h5 ini dihasilkan oleh train_cnn2d_pakcoy.py dan harus berada
# satu folder dengan app.py. Jika ada, model ini dipakai OTOMATIS di
# halaman "Deteksi Siap Panen (CNN 2D)" tanpa perlu upload manual.
CNN2D_MODEL_PATH = "pakcoy_cnn2d.h5"
CNN2D_SKLEARN_PATH = "pakcoy_cnn2d_sklearn.pkl"
CNN2D_LABELS_PATH = "class_order.json"
# Urutan default ini HARUS sama persis dengan urutan folder dataset
# (alfabetis) saat training. Skrip training akan mencetak urutan yang
# benar — salin ke sini atau ke file class_order.json jika berubah.
CNN2D_DEFAULT_LABELS = "Masa Pertumbuhan, Mendekati Panen, Panen"

# =========================================================
# 0. KONFIGURASI HALAMAN & TEMA
# =========================================================
st.set_page_config(
    page_title="Smart Farming Pakcoy",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f78166"
AMBER   = "#f0883e"
PURPLE  = "#d2a8ff"

COND_COLOR = {"Dry": ORANGE, "Optimal": GREEN, "Wet": BLUE}
STAGE_COLOR = {
    "Initial Stage": PURPLE,
    "Vegetative Stage": BLUE,
    "Pre-Harvest Stage": AMBER,
    "Harvest Ready": GREEN,
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{
        background-color: {PANEL}; border-right: 1px solid {BORDER};
    }}
    h1, h2, h3, h4, p, span, label {{ color: {TEXT}; }}
    .hero {{
        background: linear-gradient(135deg, #17301f 0%, #0d1117 100%);
        border: 1px solid #238636; border-radius: 16px;
        padding: 26px 32px; margin-bottom: 22px;
    }}
    .hero h1 {{ margin: 0; font-size: 30px; }}
    .hero p {{ color: {MUTED}; margin-top: 6px; font-size: 15px; }}
    .metric-card {{
        background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 16px 18px; text-align: center; height: 100%;
    }}
    .metric-card .icon {{ font-size: 22px; }}
    .metric-card .label {{ color: {MUTED}; font-size: 12.5px; margin: 4px 0 2px 0; }}
    .metric-card .value {{ font-size: 24px; font-weight: 700; }}
    .status-pill {{
        display: inline-block; padding: 5px 16px; border-radius: 999px;
        font-weight: 700; font-size: 13.5px; letter-spacing: .2px;
    }}
    .info-box {{
        background-color: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {BLUE};
        border-radius: 10px; padding: 14px 18px; font-size: 14px; color: {MUTED};
    }}
    .warn-box {{
        background-color: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {AMBER};
        border-radius: 10px; padding: 14px 18px; font-size: 14px; color: {MUTED};
    }}
    div[data-testid="stMetricValue"] {{ color: {BLUE}; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {BLUE} !important; }}
    hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(color=TEXT),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    legend=dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1),
    margin=dict(l=10, r=10, t=50, b=10),
)

STAGE_LABELS = {
    "Initial Stage (HST + Morphological Observation)": "Initial Stage",
    "Vegetative Stage (HST + Morphological Observation)": "Vegetative Stage",
    "Pre-Harvest Stage (HST + Morphological Observation)": "Pre-Harvest Stage",
    "Harvest Ready (HST + Morphological Observation)": "Harvest Ready",
}

STAGE_RECOMMENDATION = {
    "Initial Stage": ("🌱", "PANTAU PERTUMBUHAN AWAL",
        "Bibit pakcoy baru mulai tumbuh (±10–19 HST). Jaga kelembapan tanah tetap optimal "
        "dan pastikan sirkulasi cahaya cukup untuk mendukung fase perkecambahan."),
    "Vegetative Stage": ("🍃", "FASE PERTUMBUHAN AKTIF",
        "Tanaman memasuki fase vegetatif (±20–29 HST). Daun mulai melebar dan batang menguat. "
        "Pastikan nutrisi dan penyiraman konsisten untuk mempercepat pembentukan daun."),
    "Pre-Harvest Stage": ("⏳", "MENDEKATI MASA PANEN",
        "Tanaman berada di fase pra-panen (±30–34 HST). Kurangi penyiraman berlebih dan "
        "pantau kepadatan rimbun daun setiap hari, panen akan siap dalam waktu dekat."),
    "Harvest Ready": ("✅", "SIAP UNTUK DIPANEN",
        "Pakcoy sudah memasuki masa siap panen (≥35 HST, tingkat kematangan tinggi). "
        "Segera lakukan pemanenan agar kualitas daun tetap renyah dan segar."),
}

DATA_FILE = "Dataset_Pakcoy.xlsx"


# =========================================================
# 1. LOAD DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_excel(file)
    df["Growth Stage"] = df["Ground Truth Criteria"].map(STAGE_LABELS).fillna(df["Ground Truth Criteria"])
    return df


with st.sidebar:
    st.markdown("## 🥬 Smart Farming Pakcoy")
    st.caption("Monitoring & Klasifikasi AI untuk Budidaya Pakcoy")
    uploaded = st.file_uploader("Ganti dataset (opsional, .xlsx)", type=["xlsx"])
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Dashboard", "📊 Eksplorasi Data", "🤖 Model AI (CNN)",
         "📡 Monitoring Realtime", "📸 Deteksi Siap Panen (CNN 2D)"],
        label_visibility="collapsed",
    )

data_source = uploaded if uploaded is not None else DATA_FILE
try:
    df = load_data(data_source)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()


# =========================================================
# 2. HELPER: MODEL CNN
# =========================================================
def build_cnn(n_features, n_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=1, activation="relu", input_shape=(n_features, 1)),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource(show_spinner=False)
def train_soil_condition_model(_df, epochs=30, batch_size=16):
    d = _df.copy()
    encoder = LabelEncoder()
    d["Encode"] = encoder.fit_transform(d["Soil Condition"])
    X = d[["Soil Moisture (%)", "Temperature (°C)"]].values.astype(float)
    y = d["Encode"].values
    X = X.reshape(X.shape[0], X.shape[1], 1)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    model = build_cnn(2, len(encoder.classes_))
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test), class_weight=class_weight, verbose=0,
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    return {
        "model": model, "encoder": encoder, "history": history.history,
        "accuracy": acc, "loss": loss, "y_test": y_test, "y_pred": y_pred,
        "classes": list(encoder.classes_),
    }


@st.cache_resource(show_spinner=False)
def train_growth_stage_model(_df, epochs=30, batch_size=16):
    d = _df.copy()
    encoder = LabelEncoder()
    d["Encode"] = encoder.fit_transform(d["Growth Stage"])
    X = d[["DAP", "Ground Truth Maturity Level (%)"]].values.astype(float)
    y = d["Encode"].values
    X = X.reshape(X.shape[0], X.shape[1], 1)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    model = build_cnn(2, len(encoder.classes_))
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test), class_weight=class_weight, verbose=0,
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    return {
        "model": model, "encoder": encoder, "history": history.history,
        "accuracy": acc, "loss": loss, "y_test": y_test, "y_pred": y_pred,
        "classes": list(encoder.classes_),
    }


def predict_single(model, encoder, feat1, feat2):
    x = np.array([[feat1, feat2]], dtype=float).reshape(1, 2, 1)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return encoder.classes_[idx], probs, encoder.classes_


def metric_card(col, icon, label, value, color=BLUE):
    col.markdown(f"""
    <div class="metric-card">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value" style="color:{color}">{value}</div>
    </div>""", unsafe_allow_html=True)


def pump_logic(moisture):
    if moisture < 60:
        return "🔥 POMPA MENYALA (Tanah Kering)", ORANGE
    elif moisture <= 79:
        return "🛑 POMPA MATI (Kondisi Ideal)", GREEN
    else:
        return "❌ POMPA MATI (Tanah Terlalu Basah)", BLUE


# =========================================================
# 3. HALAMAN: DASHBOARD
# =========================================================
if page == "🏠 Dashboard":
    st.markdown(f"""
    <div class="hero">
        <h1>🥬 Smart Farming Pakcoy — Dashboard</h1>
        <p>Sistem monitoring kelembapan tanah, suhu, dan kesiapan panen tanaman Pakcoy berbasis AI (CNN).</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "📄", "Total Data", f"{len(df):,}", BLUE)
    metric_card(c2, "💧", "Rata-rata Kelembapan", f"{df['Soil Moisture (%)'].mean():.1f}%", BLUE)
    metric_card(c3, "🌡️", "Rata-rata Suhu", f"{df['Temperature (°C)'].mean():.1f}°C", ORANGE)
    metric_card(c4, "📈", "Rata-rata Kematangan", f"{df['Ground Truth Maturity Level (%)'].mean():.1f}%", GREEN)
    metric_card(c5, "🗓️", "Rentang HST (DAP)", f"{df['DAP'].min()}–{df['DAP'].max()} hari", PURPLE)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 1.4])

    with col1:
        counts = df["Soil Condition"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker=dict(colors=[COND_COLOR.get(s, MUTED) for s in counts.index],
                        line=dict(color=BG, width=2)),
            textinfo="label+percent",
        )])
        fig.update_layout(**PLOTLY_TEMPLATE, title="Distribusi Kondisi Tanah", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stage_order = ["Initial Stage", "Vegetative Stage", "Pre-Harvest Stage", "Harvest Ready"]
        stage_counts = df["Growth Stage"].value_counts().reindex(stage_order).fillna(0)
        fig2 = go.Figure(data=[go.Bar(
            x=stage_counts.index, y=stage_counts.values,
            marker_color=[STAGE_COLOR[s] for s in stage_counts.index],
        )])
        fig2.update_layout(**PLOTLY_TEMPLATE, title="Distribusi Tahap Pertumbuhan", height=360,
                            xaxis_title="", yaxis_title="Jumlah Data")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df, x="DAP", y="Ground Truth Maturity Level (%)", color="Growth Stage",
        color_discrete_map=STAGE_COLOR, opacity=0.6,
        title="Tingkat Kematangan Pakcoy terhadap Hari Setelah Tanam (DAP)",
    )
    fig3.update_layout(**PLOTLY_TEMPLATE, height=380)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 📋 Cuplikan Data")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# =========================================================
# 4. HALAMAN: EKSPLORASI DATA
# =========================================================
elif page == "📊 Eksplorasi Data":
    st.markdown("## 📊 Eksplorasi Dataset Pakcoy")

    with st.expander("🔍 Filter Data", expanded=False):
        f1, f2 = st.columns(2)
        stages_sel = f1.multiselect("Tahap Pertumbuhan", sorted(df["Growth Stage"].unique()),
                                     default=sorted(df["Growth Stage"].unique()))
        dap_range = f2.slider("Rentang DAP (Hari Setelah Tanam)",
                               int(df["DAP"].min()), int(df["DAP"].max()),
                               (int(df["DAP"].min()), int(df["DAP"].max())))
    fdf = df[df["Growth Stage"].isin(stages_sel) & df["DAP"].between(*dap_range)]

    st.markdown(f"<div class='info-box'>Menampilkan <b>{len(fdf):,}</b> dari {len(df):,} baris data sesuai filter.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    def style_cond(v):
        c = COND_COLOR.get(v, MUTED)
        return f"background-color:{c}33; color:{c}; font-weight:600"

    styled = fdf.head(200).style.map(style_cond, subset=["Soil Condition"])
    st.dataframe(styled, use_container_width=True, height=350, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.histogram(fdf, x="Soil Moisture (%)", nbins=20,
                            color_discrete_sequence=[BLUE])
        fig.add_vline(x=60, line_dash="dash", line_color=ORANGE, annotation_text="Batas Kering")
        fig.add_vline(x=79, line_dash="dash", line_color=GREEN, annotation_text="Batas Basah")
        fig.update_layout(**PLOTLY_TEMPLATE, title="Distribusi Kelembapan Tanah", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(fdf, x="Soil Moisture (%)", y="Temperature (°C)",
                           color="Soil Condition", color_discrete_map=COND_COLOR, opacity=0.55)
        fig2.update_layout(**PLOTLY_TEMPLATE, title="Kelembapan vs Suhu", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        fig3 = px.box(fdf, x="Growth Stage", y="Temperature (°C)",
                       color="Growth Stage", color_discrete_map=STAGE_COLOR,
                       category_orders={"Growth Stage": ["Initial Stage", "Vegetative Stage",
                                                          "Pre-Harvest Stage", "Harvest Ready"]})
        fig3.update_layout(**PLOTLY_TEMPLATE, title="Suhu per Tahap Pertumbuhan", height=340, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "<div class='warn-box'>⚠️ Kondisi tanah <b>Dry</b> dan <b>Wet</b> jumlahnya sangat sedikit "
        "dibanding <b>Optimal</b> pada dataset ini (data tidak seimbang). Perhatikan hal ini saat "
        "menilai akurasi model klasifikasi kondisi tanah.</div>", unsafe_allow_html=True)


# =========================================================
# 5. HALAMAN: MODEL AI (CNN)
# =========================================================
elif page == "🤖 Model AI (CNN)":
    st.markdown("## 🤖 Model AI — Klasifikasi CNN")
    tab1, tab2 = st.tabs(["💧 Kondisi Tanah", "🌱 Tahap Pertumbuhan & Kesiapan Panen"])

    # ---------------- TAB 1: SOIL CONDITION ----------------
    with tab1:
        st.markdown(
            "<div class='info-box'>Model CNN 1D memprediksi <b>Kondisi Tanah</b> (Dry / Optimal / Wet) "
            "dari fitur <b>Kelembapan Tanah</b> dan <b>Suhu</b>, mengikuti arsitektur dari sistem "
            "Bayam Brazil sebelumnya.</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Latih / Muat Model Kondisi Tanah", key="train_soil"):
            with st.spinner("Melatih model CNN (30 epoch)..."):
                st.session_state["soil_result"] = train_soil_condition_model(df)

        if "soil_result" in st.session_state:
            res = st.session_state["soil_result"]
            hist = res["history"]

            c1, c2, c3 = st.columns(3)
            metric_card(c1, "🎯", "Akurasi Akhir (Test)", f"{res['accuracy']*100:.2f}%", BLUE)
            metric_card(c2, "⭐", "Best Val Accuracy", f"{max(hist['val_accuracy'])*100:.2f}%", GREEN)
            metric_card(c3, "📉", "Final Loss", f"{hist['loss'][-1]:.4f}", ORANGE)
            st.markdown("<br>", unsafe_allow_html=True)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Kurva Akurasi", "Kurva Loss"))
            ep = list(range(1, len(hist["accuracy"]) + 1))
            fig.add_trace(go.Scatter(x=ep, y=hist["accuracy"], name="Train Acc", line=dict(color=BLUE)), 1, 1)
            fig.add_trace(go.Scatter(x=ep, y=hist["val_accuracy"], name="Val Acc", line=dict(color=GREEN, dash="dash")), 1, 1)
            fig.add_trace(go.Scatter(x=ep, y=hist["loss"], name="Train Loss", line=dict(color=ORANGE)), 1, 2)
            fig.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="Val Loss", line=dict(color=PURPLE, dash="dash")), 1, 2)
            fig.update_layout(**PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig, use_container_width=True)

            cm = confusion_matrix(res["y_test"], res["y_pred"])
            fig_cm = px.imshow(cm, text_auto=True, x=res["classes"], y=res["classes"],
                                color_continuous_scale="Blues",
                                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
            fig_cm.update_layout(**PLOTLY_TEMPLATE, title="Confusion Matrix", height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown(
                "<div class='warn-box'>⚠️ Karena kelas <b>Dry</b> dan <b>Wet</b> sangat sedikit di dataset, "
                "skor pada kelas tersebut kurang bisa diandalkan meski sudah memakai <i>class weighting</i>. "
                "Tambah data sensor riil untuk kondisi kering/basah agar model lebih andal.</div>",
                unsafe_allow_html=True)

            st.markdown("#### 🧪 Coba Prediksi Manual")
            p1, p2, p3 = st.columns([1, 1, 1])
            m_val = p1.slider("Kelembapan Tanah (%)", 40, 90, 70)
            t_val = p2.slider("Suhu (°C)", 22.0, 33.0, 27.0, step=0.1)
            if p3.button("🔮 Prediksi Kondisi Tanah"):
                label, probs, classes = predict_single(res["model"], res["encoder"], m_val, t_val)
                color = COND_COLOR.get(label, BLUE)
                st.markdown(f"<span class='status-pill' style='background:{color}33;color:{color}'>"
                            f"Hasil: {label}</span>", unsafe_allow_html=True)
                prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": probs})
                fig_p = px.bar(prob_df, x="Kelas", y="Probabilitas", color="Kelas",
                                color_discrete_map=COND_COLOR, range_y=[0, 1])
                fig_p.update_layout(**PLOTLY_TEMPLATE, height=300, showlegend=False)
                st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Klik tombol di atas untuk melatih model (hasil akan tersimpan di sesi ini).")

    # ---------------- TAB 2: GROWTH STAGE ----------------
    with tab2:
        st.markdown(
            "<div class='info-box'>Model CNN 1D memprediksi <b>Tahap Pertumbuhan / Kesiapan Panen</b> "
            "dari fitur <b>DAP (Hari Setelah Tanam)</b> dan <b>Tingkat Kematangan</b>. Bagian ini menggantikan "
            "modul klasifikasi citra daun pada sistem Bayam Brazil, karena dataset Pakcoy berbasis sensor/HST.</div>",
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Latih / Muat Model Tahap Pertumbuhan", key="train_stage"):
            with st.spinner("Melatih model CNN (30 epoch)..."):
                st.session_state["stage_result"] = train_growth_stage_model(df)

        if "stage_result" in st.session_state:
            res = st.session_state["stage_result"]
            hist = res["history"]

            c1, c2, c3 = st.columns(3)
            metric_card(c1, "🎯", "Akurasi Akhir (Test)", f"{res['accuracy']*100:.2f}%", BLUE)
            metric_card(c2, "⭐", "Best Val Accuracy", f"{max(hist['val_accuracy'])*100:.2f}%", GREEN)
            metric_card(c3, "📉", "Final Loss", f"{hist['loss'][-1]:.4f}", ORANGE)
            st.markdown("<br>", unsafe_allow_html=True)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Kurva Akurasi", "Kurva Loss"))
            ep = list(range(1, len(hist["accuracy"]) + 1))
            fig.add_trace(go.Scatter(x=ep, y=hist["accuracy"], name="Train Acc", line=dict(color=BLUE)), 1, 1)
            fig.add_trace(go.Scatter(x=ep, y=hist["val_accuracy"], name="Val Acc", line=dict(color=GREEN, dash="dash")), 1, 1)
            fig.add_trace(go.Scatter(x=ep, y=hist["loss"], name="Train Loss", line=dict(color=ORANGE)), 1, 2)
            fig.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="Val Loss", line=dict(color=PURPLE, dash="dash")), 1, 2)
            fig.update_layout(**PLOTLY_TEMPLATE, height=380)
            st.plotly_chart(fig, use_container_width=True)

            cm = confusion_matrix(res["y_test"], res["y_pred"])
            fig_cm = px.imshow(cm, text_auto=True, x=res["classes"], y=res["classes"],
                                color_continuous_scale="Greens",
                                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
            fig_cm.update_layout(**PLOTLY_TEMPLATE, title="Confusion Matrix", height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("#### 🧪 Coba Prediksi Manual")
            p1, p2, p3 = st.columns([1, 1, 1])
            dap_val = p1.slider("DAP (Hari Setelah Tanam)", int(df["DAP"].min()), int(df["DAP"].max()), 30)
            mat_val = p2.slider("Tingkat Kematangan (%)", 30, 100, 60)
            if p3.button("🔮 Prediksi Tahap Pertumbuhan"):
                label, probs, classes = predict_single(res["model"], res["encoder"], dap_val, mat_val)
                icon, title, desc = STAGE_RECOMMENDATION.get(label, ("🌿", label, ""))
                color = STAGE_COLOR.get(label, BLUE)
                st.markdown(f"<span class='status-pill' style='background:{color}33;color:{color}'>"
                            f"{icon} {title}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-box' style='margin-top:10px'>{desc}</div>", unsafe_allow_html=True)
                prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": probs})
                fig_p = px.bar(prob_df, x="Kelas", y="Probabilitas", color="Kelas",
                                color_discrete_map=STAGE_COLOR, range_y=[0, 1])
                fig_p.update_layout(**PLOTLY_TEMPLATE, height=300, showlegend=False)
                st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Klik tombol di atas untuk melatih model (hasil akan tersimpan di sesi ini).")


# =========================================================
# 6. HALAMAN: MONITORING REALTIME (SIMULASI)
# =========================================================
elif page == "📡 Monitoring Realtime":
    st.markdown("## 📡 Monitoring Realtime — Simulasi Sensor IoT Pakcoy")
    st.markdown(
        "<div class='info-box'>Mensimulasikan pembacaan sensor kelembapan & suhu setiap beberapa detik, "
        "diklasifikasikan oleh model CNN, dilengkapi tren regresi linear (LSCM) dan logika otomatisasi pompa air.</div>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "soil_result" not in st.session_state:
        with st.spinner("Menyiapkan model klasifikasi kondisi tanah..."):
            st.session_state["soil_result"] = train_soil_condition_model(df)
    soil_res = st.session_state["soil_result"]

    if "rt_history" not in st.session_state:
        st.session_state["rt_history"] = df["Soil Moisture (%)"].iloc[-100:].tolist()
        st.session_state["rt_counter"] = 0

    m_min, m_max = float(df["Soil Moisture (%)"].min()), float(df["Soil Moisture (%)"].max())
    t_min, t_max = float(df["Temperature (°C)"].min()), float(df["Temperature (°C)"].max())

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    steps = ctrl1.number_input("Jumlah langkah simulasi", 5, 100, 20)
    speed = ctrl2.slider("Jeda antar-langkah (detik)", 0.1, 2.0, 0.4)
    run = ctrl3.button("▶️ Jalankan Simulasi", type="primary")
    reset = ctrl3.button("🔄 Reset Riwayat")

    if reset:
        st.session_state["rt_history"] = df["Soil Moisture (%)"].iloc[-100:].tolist()
        st.session_state["rt_counter"] = 0
        st.rerun()

    chart_ph = st.empty()
    cards_ph = st.empty()

    def render_frame(history, moisture, temp, cnn_label, pump_text, pump_color, trend, counter):
        x = list(range(len(history)))
        x_arr = np.array(x).reshape(-1, 1)
        lscm = LinearRegression().fit(x_arr, history).predict(x_arr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=history, mode="lines", name="Sensor Realtime",
                                  line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=x, y=lscm, mode="lines", name="Tren LSCM",
                                  line=dict(color=AMBER, width=2, dash="dash")))
        fig.add_hline(y=60, line_dash="dot", line_color=ORANGE, annotation_text="Batas Kering")
        fig.add_hline(y=80, line_dash="dot", line_color=GREEN, annotation_text="Batas Basah")
        fig.update_layout(**PLOTLY_TEMPLATE, height=380,
                           title=f"📡 Monitoring Kelembapan Realtime — Log ke-{counter}",
                           yaxis_title="Kelembapan Tanah (%)", xaxis_title="Data ke-n (100 terakhir)")
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"chart_{counter}")

        cond_color = COND_COLOR.get(cnn_label, BLUE)
        with cards_ph.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            metric_card(c1, "💧", "Kelembapan Tanah", f"{moisture}%", BLUE)
            metric_card(c2, "🌡️", "Suhu Udara", f"{temp}°C", ORANGE)
            metric_card(c3, "🤖", "CNN Klasifikasi", cnn_label, cond_color)
            metric_card(c4, "📈", "Tren LSCM", f"{trend:.1f}%", PURPLE)
            metric_card(c5, "⚡", "Pompa Air", pump_text.split(" ")[0], pump_color)
            st.markdown(f"<div class='info-box' style='margin-top:10px'>{pump_text}</div>", unsafe_allow_html=True)

    if run:
        history = st.session_state["rt_history"]
        counter = st.session_state["rt_counter"]
        for _ in range(int(steps)):
            counter += 1
            moisture = int(np.random.randint(int(m_min), int(m_max) + 1))
            temp = round(float(np.random.uniform(t_min, t_max)), 1)

            history.append(moisture)
            if len(history) > 100:
                history.pop(0)

            x_arr = np.array(range(len(history))).reshape(-1, 1)
            trend = LinearRegression().fit(x_arr, history).predict(x_arr)[-1]

            cnn_label, probs, _ = predict_single(soil_res["model"], soil_res["encoder"], moisture, temp)
            pump_text, pump_color = pump_logic(moisture)

            render_frame(history, moisture, temp, cnn_label, pump_text, pump_color, trend, counter)
            time.sleep(speed)

        st.session_state["rt_history"] = history
        st.session_state["rt_counter"] = counter
    else:
        # tampilkan status terakhir tanpa animasi
        history = st.session_state["rt_history"]
        moisture = history[-1]
        temp = round(float(df["Temperature (°C)"].iloc[-1]), 1)
        x_arr = np.array(range(len(history))).reshape(-1, 1)
        trend = LinearRegression().fit(x_arr, history).predict(x_arr)[-1]
        cnn_label, probs, _ = predict_single(soil_res["model"], soil_res["encoder"], moisture, temp)
        pump_text, pump_color = pump_logic(moisture)
        render_frame(history, moisture, temp, cnn_label, pump_text, pump_color, trend, st.session_state["rt_counter"])


# =========================================================
# 7. HALAMAN: DETEKSI SIAP PANEN DARI FOTO (CNN 2D)
# =========================================================
elif page == "📸 Deteksi Siap Panen (CNN 2D)":
    st.markdown("## 📸 Deteksi Kesiapan Panen dari Foto Daun Pakcoy")
    st.markdown(
        "<div class='info-box'>Unggah foto tanaman Pakcoy untuk memperkirakan tahap kesiapan panen "
        "berdasarkan analisis visual daun — warna, kepadatan rimbun, dan cakupan area dalam bingkai foto.</div>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Muat model bawaan. Dua tingkat:
    #  1) pakcoy_cnn2d.h5      — CNN Keras asli (kalau TensorFlow & file-nya ada)
    #  2) pakcoy_cnn2d_sklearn.pkl — classifier fitur warna/tekstur, sudah
    #     dilatih sungguhan dari foto asli kebun (belum_panen, mendekati_panen)
    #     lewat train_sklearn.py — dipakai kalau .h5 tidak ada.
    #  3) fallback heuristik lama (kalau dua-duanya tidak ada)
    # -----------------------------------------------------
    @st.cache_resource(show_spinner="Memuat model CNN 2D...")
    def load_bundled_cnn2d(path):
        if not os.path.exists(path):
            return None
        return tf.keras.models.load_model(path)

    @st.cache_resource(show_spinner="Memuat model klasifikasi...")
    def load_bundled_sklearn(path):
        if not os.path.exists(path):
            return None
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_bundled_labels(default_labels):
        if os.path.exists(CNN2D_LABELS_PATH):
            try:
                import json
                with open(CNN2D_LABELS_PATH) as f:
                    return ", ".join(json.load(f))
            except Exception:
                pass
        return default_labels

    bundled_cnn = load_bundled_cnn2d(CNN2D_MODEL_PATH)
    bundled_sklearn = load_bundled_sklearn(CNN2D_SKLEARN_PATH)
    bundled_labels_default = load_bundled_labels(CNN2D_DEFAULT_LABELS)

    with st.expander("⚙️ Model deteksi foto", expanded=(bundled_cnn is None and bundled_sklearn is None)):
        if bundled_cnn is not None:
            st.success(
                f"Model CNN 2D bawaan **{CNN2D_MODEL_PATH}** aktif "
                f"(kelas: {bundled_labels_default}). Prediksi otomatis memakai model ini."
            )
        elif bundled_sklearn is not None:
            cls_disp = ", ".join(c.replace("_", " ").title() for c in bundled_sklearn["classes"])
            st.success(
                f"Model klasifikasi bawaan **{CNN2D_SKLEARN_PATH}** aktif — dilatih dari foto asli "
                f"kebun (kelas: {cls_disp}). Ini bukan CNN piksel-mentah (dilatih dengan fitur "
                "warna/tekstur karena lingkungan training tidak punya TensorFlow), tapi hasilnya "
                "model terlatih sungguhan, bukan aturan hardcode."
            )
            st.caption(
                "Untuk CNN 2D asli yang lebih akurat: jalankan `python train_cnn2d_pakcoy.py` di "
                "komputer yang punya TensorFlow, taruh `pakcoy_cnn2d.h5` di folder ini — otomatis "
                "menggantikan model ini."
            )
        else:
            st.warning(
                "Belum ada model terlatih di folder aplikasi, jadi mode bawaan di bawah memakai "
                "analisis warna & ukuran daun (heuristik)."
            )
        st.caption("Bisa juga unggah model CNN 2D lain (format .h5, input 150x150) untuk override sementara:")
        model_file = st.file_uploader("Model CNN 2D (.h5) — opsional, override", type=["h5"], key="cnn2d_model")
        label_names_input = st.text_input(
            "Urutan label kelas sesuai output softmax model (pisahkan koma)",
            value=bundled_labels_default)

    uploaded_img = st.file_uploader("Unggah Foto Daun Pakcoy", type=["png", "jpg", "jpeg"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img).convert("RGB")
        col_img, col_result = st.columns([1, 1.3])
        col_img.image(img, caption="Foto yang diunggah", use_container_width=True)

        probs, labels = None, None

        active_cnn, active_sklearn, mode_source = None, None, None
        if model_file is not None:
            with open("_cnn2d_uploaded_model.h5", "wb") as f:
                f.write(model_file.read())
            active_cnn = tf.keras.models.load_model("_cnn2d_uploaded_model.h5")
            mode_source = "Model CNN 2D terlatih (diunggah pengguna, override)"
        elif bundled_cnn is not None:
            active_cnn = bundled_cnn
            mode_source = "Model CNN 2D terlatih (bawaan aplikasi)"
        elif bundled_sklearn is not None:
            active_sklearn = bundled_sklearn
            mode_source = "Model klasifikasi terlatih — fitur warna/tekstur (bawaan aplikasi)"

        if active_cnn is not None:
            labels = [s.strip() for s in label_names_input.split(",") if s.strip()]
            img_resized = img.resize((150, 150))
            arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            probs = active_cnn.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = labels[idx] if idx < len(labels) else f"Kelas {idx}"
            mode_text = mode_source
        elif active_sklearn is not None:
            from pakcoy_features import extract_features
            feat = extract_features(img).reshape(1, -1)
            probs = active_sklearn["model"].predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            labels = [c.replace("_", " ").title() for c in active_sklearn["classes"]]
            label = labels[idx] if idx < len(labels) else f"Kelas {idx}"
            mode_text = mode_source
        else:
            arr = np.array(img.resize((300, 300))).astype(float)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            green_mask = (g > r) & (g > b * 0.9) & (g > 40)
            coverage = float(green_mask.mean())
            if coverage > 0.02:
                sat = float((((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) / 255.0))[green_mask].mean())
                green_level = float(g[green_mask].mean() / 255.0)
            else:
                sat, green_level = 0.0, 0.0
            score = 0.5 * coverage + 0.3 * sat + 0.2 * green_level

            if coverage < 0.15 or score < 0.28:
                label = "Belum Panen"
            elif score < 0.42:
                label = "Mendekati Panen"
            else:
                label = "Siap Panen"
            mode_text = "Estimasi analisis warna & ukuran (heuristik, bukan CNN terlatih)"

        icon_map = {
            "Belum Panen": ("❌", ORANGE),
            "Mendekati Panen": ("⏳", AMBER),
            "Siap Panen": ("🍃", GREEN),
        }
        icon, color = icon_map.get(label, ("🌱", BLUE))

        with col_result:
            st.markdown(f"<span class='status-pill' style='background:{color}33;color:{color}'>"
                        f"{icon} {label}</span>", unsafe_allow_html=True)
            st.caption(f"Metode: {mode_text}")

            rekom = {
                "Belum Panen": "Daun bagian atas masih menguncup/sempit dengan warna hijau pucat. "
                                "Tanaman masih renggang dalam pot — belum disarankan dipanen.",
                "Mendekati Panen": "Lebar daun mulai berkembang dan warna hijau mulai normal/segar. "
                                    "Pantau berkala, kemungkinan siap panen dalam 1–2 minggu ke depan.",
                "Siap Panen": "Daun sudah melebar, warna hijau segar pekat, dan kondisi rimbun penuh "
                              "menutupi pot. Disarankan segera dipanen agar kualitas daun tetap renyah.",
            }
            st.markdown(f"<div class='info-box' style='margin-top:10px'>{rekom.get(label, '')}</div>",
                        unsafe_allow_html=True)

            if probs is not None:
                prob_df = pd.DataFrame({"Kelas": labels[:len(probs)], "Probabilitas": probs})
                fig_p = px.bar(prob_df, x="Kelas", y="Probabilitas", range_y=[0, 1],
                                color_discrete_sequence=[GREEN])
            else:
                metric_df = pd.DataFrame({
                    "Indikator": ["Cakupan Area Daun", "Saturasi Warna", "Tingkat Kehijauan"],
                    "Skor (0-1)": [round(coverage, 2), round(sat, 2), round(green_level, 2)],
                })
                fig_p = px.bar(metric_df, x="Indikator", y="Skor (0-1)", range_y=[0, 1],
                                color_discrete_sequence=[GREEN])
            fig_p.update_layout(**PLOTLY_TEMPLATE, height=280, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

        if active_cnn is None and active_sklearn is None:
            st.markdown(
                "<div class='warn-box'>⚠️ Mode analisis warna adalah pendekatan sederhana untuk demo/prototipe, "
                "bukan hasil deep learning. Untuk akurasi lebih baik, latih model memakai dataset foto daun "
                "Pakcoy berlabel (masa_pertumbuhan / mendekati_panen / panen), lalu jalankan "
                "train_sklearn.py atau train_cnn2d_pakcoy.py.</div>", unsafe_allow_html=True)
    else:
        st.info("Unggah foto daun Pakcoy di atas untuk memulai deteksi.")
