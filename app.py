"""
Dashboard Segmentasi UMKM Kabupaten Sukabumi
=============================================
Capstone Project - Magister Informatika (Data Science), Universitas Nusa Putra

Pipeline mengikuti artikel "Performance Comparison of K-Means, DBSCAN, and
Hierarchical Clustering Algorithms for MSME Segmentation in Sukabumi Regency" (rev 2):
  2.057 data -> buang Asset/Omset kosong/0 -> buang 57 data salah satuan -> 1.771 data
  log1p -> StandardScaler (satu scaler untuk semua algoritma)
  K-Means (K=4) vs Hierarchical Ward (K=4) vs DBSCAN (eps=0,41; MinPts=5)
  Feasibility gate (cluster terkecil >= 5%, noise <= 10%) + skor komposit berbobot

Jalankan:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# =============================================================================
# KONFIGURASI
# =============================================================================
st.set_page_config(
    page_title="Dashboard Segmentasi UMKM Sukabumi",
    page_icon="📊",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "UMKM 2057.xlsx"
RANDOM_STATE = 42
K_FINAL = 4
DB_EPS = 0.41
DB_MINPTS = 5
MIN_SHARE = 0.05          # feasibility gate: cluster terkecil >= 5% data
MAX_NOISE = 0.10          # feasibility gate: noise <= 10% data
UNIT_ERR_ASSET = 5_000    # data salah satuan: Asset <= Rp5.000 dan Omset <= Rp9.000
UNIT_ERR_OMSET = 9_000
BOBOT_DEFAULT = {"Silhouette": 0.30, "DBI": 0.30, "CHI": 0.20, "Balance": 0.20}

# Skenario sensitivitas bobot (Tabel III artikel)
SKENARIO_BOBOT = {
    "Original (30/30/20/20)": (0.30, 0.30, 0.20, 0.20),
    "Equal (25/25/25/25)": (0.25, 0.25, 0.25, 0.25),
    "Balance-heavy (20/20/20/40)": (0.20, 0.20, 0.20, 0.40),
    "Sil-heavy (50/20/15/15)": (0.50, 0.20, 0.15, 0.15),
    "DBI/CHI-heavy (20/40/30/10)": (0.20, 0.40, 0.30, 0.10),
    "Extreme (0/0/0/100)": (0.0, 0.0, 0.0, 1.0),
}

# Profil segmen. Warna sudah divalidasi aman untuk buta warna (dengan
# penanda bentuk sebagai encoding kedua pada scatter).
SEGMEN = {
    "dagang": {
        "no": 1,
        "nama": "UMKM Dagang Omset Tinggi",
        "en": "High-Revenue Trading MSMEs",
        "warna": "#eda100",
        "simbol": "square",
        "ciri": "Aset sedang tetapi omset sangat tinggi — pola usaha dagang/reseller.",
        "rekomendasi": [
            "Pembiayaan perdagangan (trade financing) dan modal kerja berbasis omset",
            "Pendampingan digitalisasi: onboarding e-commerce dan marketplace",
            "Penguatan rantai pasok dan kemitraan dengan pemasok",
        ],
    },
    "berkembang": {
        "no": 2,
        "nama": "Usaha Mikro Berkembang",
        "en": "Growing Micro Enterprises",
        "warna": "#1baf7a",
        "simbol": "circle",
        "ciri": "Aset dan omset seimbang pada level menengah — segmen terbesar.",
        "rekomendasi": [
            "Kredit mikro modal kerja (mis. KUR Mikro)",
            "Pembiayaan berbasis koperasi",
            "Pelatihan manajerial dan pemasaran untuk mempercepat pertumbuhan",
        ],
    },
    "mapan": {
        "no": 3,
        "nama": "UMKM Mapan",
        "en": "Established MSMEs",
        "warna": "#2a78d6",
        "simbol": "diamond",
        "ciri": "Aset dan omset tertinggi — potensi terbesar untuk naik kelas.",
        "rekomendasi": [
            "Program naik kelas menjadi usaha kecil/menengah",
            "Akses pembiayaan perbankan formal dengan plafon lebih besar",
            "Sertifikasi produk, standardisasi, dan perluasan pasar",
        ],
    },
    "subsisten": {
        "no": 4,
        "nama": "Usaha Mikro Subsisten",
        "en": "Subsistence Micro Enterprises",
        "warna": "#e87ba4",
        "simbol": "triangle-up",
        "ciri": "Aset dan omset terendah — segmen paling rentan.",
        "rekomendasi": [
            "Hibah/bantuan modal mikro (micro-grant)",
            "Literasi keuangan dasar dan pembukuan sederhana",
            "Pendampingan legalitas usaha (NIB) dan registrasi",
        ],
    },
}
URUTAN_SEGMEN = sorted(SEGMEN, key=lambda k: SEGMEN[k]["no"])
LABEL_SEGMEN = {k: f"Cluster {v['no']} · {v['nama']}" for k, v in SEGMEN.items()}
WARNA_LABEL = {LABEL_SEGMEN[k]: SEGMEN[k]["warna"] for k in SEGMEN}
SIMBOL_LABEL = {LABEL_SEGMEN[k]: SEGMEN[k]["simbol"] for k in SEGMEN}
URUTAN_LABEL = [LABEL_SEGMEN[k] for k in URUTAN_SEGMEN]
WARNA_ALGO = {"K-Means": "#2a78d6", "Hierarchical": "#1baf7a", "DBSCAN": "#eda100"}

# Kompatibilitas parameter lebar antar versi Streamlit
_ver = tuple(int(p) for p in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}


# =============================================================================
# UTILITAS FORMAT
# =============================================================================
def angka(x, des=0):
    """Format angka gaya Indonesia: 1.234.567,89"""
    s = f"{x:,.{des}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def rp(x):
    """Rupiah ringkas: Rp 1,3 jt / Rp 2,4 M"""
    if pd.isna(x):
        return "-"
    for batas, akhiran in [(1e12, "T"), (1e9, "M"), (1e6, "jt"), (1e3, "rb")]:
        if abs(x) >= batas:
            return f"Rp {angka(x / batas, 1)} {akhiran}"
    return f"Rp {angka(x)}"


def rp_penuh(x):
    return "-" if pd.isna(x) else f"Rp {angka(x)}"


LOG_TICKS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
LOG_TEXT = ["100", "1rb", "10rb", "100rb", "1jt", "10jt", "100jt", "1M", "10M"]


def sumbu_log(fig, x=True, y=True):
    if x:
        fig.update_xaxes(type="log", tickvals=LOG_TICKS, ticktext=LOG_TEXT)
    if y:
        fig.update_yaxes(type="log", tickvals=LOG_TICKS, ticktext=LOG_TEXT)
    return fig


def rapikan(fig, tinggi=420, legenda=True):
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=110 if legenda else 50, b=10),
        title=dict(y=0.985, yanchor="top", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title=None)
        if legenda else None,
        showlegend=legenda,
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


def tampil(fig):
    st.plotly_chart(fig, **WIDE)


def tabel(df, **kw):
    st.dataframe(df, hide_index=True, **WIDE, **kw)


# =============================================================================
# PIPELINE DATA & MODEL (di-cache)
# =============================================================================
def _norm_gender(x):
    if pd.isna(x):
        return "Tidak diketahui"
    s = str(x).strip().lower().replace(" ", "")
    if s.startswith("p"):
        return "Perempuan"
    if s.startswith("l"):
        return "Laki-laki"
    return "Tidak diketahui"


@st.cache_data(show_spinner="Membaca data UMKM...")
def baca_data(file_bytes: bytes) -> pd.DataFrame:
    import io

    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner="Membersihkan data dan melatih model...")
def jalankan_pipeline(raw: pd.DataFrame):
    df = raw.copy()
    n_awal = len(df)

    # 1. Duplikat identik
    kolom_id = [c for c in ["Nama Lengkap", "Jenis Kelamin", "Usia", "Nama Usaha",
                            "Alamat", "Kecamatan", "Produksi", "Asset", "Omset"]
                if c in df.columns]
    n_dup = int(df.duplicated(subset=kolom_id).sum())
    df = df.drop_duplicates(subset=kolom_id).copy()

    # 2. Asset / Omset kosong, non-numerik, atau 0
    df["Asset"] = pd.to_numeric(df["Asset"], errors="coerce")
    df["Omset"] = pd.to_numeric(df["Omset"], errors="coerce")
    valid = (df["Asset"] > 0) & (df["Omset"] > 0)
    n_invalid = int((~valid).sum())
    df = df[valid].copy()

    # 3. Data salah satuan (nilai ribuan rupiah yang tidak masuk akal)
    salah = (df["Asset"] <= UNIT_ERR_ASSET) & (df["Omset"] <= UNIT_ERR_OMSET)
    df_salah = df[salah].copy()
    df = df[~salah].reset_index(drop=True)

    # Rapikan kolom kategorikal untuk tampilan
    df["Kecamatan"] = (df.get("Kecamatan", pd.Series(dtype=str)).fillna("Tidak diketahui")
                       .astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title())
    df_salah["Kecamatan"] = df_salah["Kecamatan"].astype(str).str.strip().str.title()
    if "Jenis Kelamin" in df.columns:
        df["Jenis Kelamin"] = df["Jenis Kelamin"].map(_norm_gender)
    if "Pelatihan yang di ikuti" in df.columns:
        df["Pelatihan yang di ikuti"] = (df["Pelatihan yang di ikuti"].fillna("-").astype(str)
                                         .str.strip().str.replace(r"\s+", " ", regex=True))
    for c in ["Nama Usaha", "Produksi"]:
        if c in df.columns:
            df[c] = df[c].fillna("-").astype(str).str.strip()

    funnel = pd.DataFrame({
        "Tahap": ["Data awal", "Duplikat identik", "Asset/Omset kosong, non-numerik, atau 0",
                  "Salah satuan (Asset ≤ Rp5.000 & Omset ≤ Rp9.000)", "Data siap analisis"],
        "Jumlah": [n_awal, -n_dup, -n_invalid, -len(df_salah), len(df)],
    })

    # 4. Transformasi: log1p -> StandardScaler (satu scaler untuk semua algoritma)
    X_log = np.log1p(df[["Asset", "Omset"]].values)
    scaler = StandardScaler().fit(X_log)
    X = scaler.transform(X_log)

    # 5. Model
    kmeans = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10).fit(X)
    lab_km = kmeans.labels_
    lab_hc = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward").fit_predict(X)
    lab_db = DBSCAN(eps=DB_EPS, min_samples=DB_MINPTS).fit_predict(X)

    # 6. Penamaan segmen berdasarkan karakteristik centroid (tidak bergantung
    #    pada nomor label acak K-Means)
    pusat = kmeans.cluster_centers_
    skala = pusat.sum(axis=1)                   # skala ekonomi (asset + omset)
    rasio = pusat[:, 1] - pusat[:, 0]           # dominasi omset terhadap aset
    idx_mapan = int(np.argmax(skala))
    idx_sub = int(np.argmin(skala))
    sisa = [i for i in range(K_FINAL) if i not in (idx_mapan, idx_sub)]
    idx_dagang = max(sisa, key=lambda i: rasio[i])
    idx_berk = [i for i in sisa if i != idx_dagang][0]
    peta = {idx_mapan: "mapan", idx_sub: "subsisten", idx_dagang: "dagang", idx_berk: "berkembang"}

    df["Segmen_Key"] = [peta[l] for l in lab_km]
    df["Cluster"] = df["Segmen_Key"].map(lambda k: SEGMEN[k]["no"])
    df["Segmen"] = df["Segmen_Key"].map(LABEL_SEGMEN)
    df["Cluster_HC"] = lab_hc
    df["Cluster_DBSCAN"] = lab_db

    centroid = pd.DataFrame(np.expm1(scaler.inverse_transform(pusat)), columns=["Asset", "Omset"])
    centroid["Segmen_Key"] = [peta[i] for i in range(K_FINAL)]
    centroid["Segmen"] = centroid["Segmen_Key"].map(LABEL_SEGMEN)

    return {
        "df": df, "df_salah": df_salah, "funnel": funnel, "X": X, "X_log": X_log,
        "scaler": scaler, "kmeans": kmeans, "peta": peta, "centroid": centroid,
        "lab_km": lab_km, "lab_hc": lab_hc, "lab_db": lab_db,
    }


def metrik(X, label, n_total=None):
    """Silhouette, DBI, CHI, ukuran cluster, balance, dan feasibility gate."""
    label = np.asarray(label)
    n_total = n_total or len(label)
    mask = label != -1
    n_noise = int((~mask).sum())
    ukuran = np.bincount(label[mask]) if mask.any() else np.array([0])
    ukuran = ukuran[ukuran > 0]
    n_cl = len(ukuran)
    hasil = {"Cluster": n_cl, "Noise": n_noise,
             "Silhouette": np.nan, "DBI": np.nan, "CHI": np.nan,
             "Terkecil": int(ukuran.min()) if n_cl else 0,
             "Terbesar": int(ukuran.max()) if n_cl else 0}
    if n_cl >= 2:
        hasil["Silhouette"] = silhouette_score(X[mask], label[mask])
        hasil["DBI"] = davies_bouldin_score(X[mask], label[mask])
        hasil["CHI"] = calinski_harabasz_score(X[mask], label[mask])
    hasil["Balance"] = hasil["Terkecil"] / hasil["Terbesar"] if hasil["Terbesar"] else 0
    hasil["Lolos Gate"] = bool(hasil["Terkecil"] >= MIN_SHARE * n_total
                               and n_noise <= MAX_NOISE * n_total and n_cl >= 2)
    return hasil


def skor_komposit(tabel_m, bobot):
    """Min-max antar kandidat; DBI dibalik (1 - DBI_norm); Balance dipakai langsung."""
    t = tabel_m.copy()

    def mm(s):
        rentang = s.max() - s.min()
        return pd.Series(1.0, index=s.index) if rentang == 0 else (s - s.min()) / rentang

    ws, wd, wc, wb = bobot
    t["Sil_norm"] = mm(t["Silhouette"])
    t["DBI_norm"] = 1 - mm(t["DBI"]) if t["DBI"].max() != t["DBI"].min() else 1.0
    t["CHI_norm"] = mm(t["CHI"])
    t["Skor"] = ws * t["Sil_norm"] + wd * t["DBI_norm"] + wc * t["CHI_norm"] + wb * t["Balance"]
    return t


@st.cache_data(show_spinner="Mengevaluasi K = 2 sampai 10...")
def evaluasi_k(X):
    baris = []
    for k in range(2, 11):
        lab = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        m = metrik(X, lab.labels_)
        m.update({"K": k, "WCSS": lab.inertia_,
                  "Ukuran": ", ".join(str(v) for v in np.bincount(lab.labels_))})
        baris.append(m)
    t = pd.DataFrame(baris)
    t = skor_komposit(t, tuple(BOBOT_DEFAULT.values()))
    return t


@st.cache_data(show_spinner=False)
def k_distance(X, k):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, k - 1])


@st.cache_data(show_spinner="Grid search DBSCAN (ε 0,05–1,00; MinPts 5/10/15/20)...")
def grid_dbscan(X):
    baris = []
    for mp in [5, 10, 15, 20]:
        for eps in np.round(np.arange(0.05, 1.001, 0.01), 2):
            lab = DBSCAN(eps=eps, min_samples=mp).fit_predict(X)
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            n_noise = int((lab == -1).sum())
            if 2 <= n_cl <= 10 and n_noise <= MAX_NOISE * len(X):
                m = metrik(X, lab)
                m.update({"eps": eps, "MinPts": mp})
                baris.append(m)
    t = pd.DataFrame(baris)
    if len(t):
        t = skor_komposit(t, tuple(BOBOT_DEFAULT.values()))
    return t


@st.cache_data(show_spinner=False)
def matriks_linkage(X):
    return linkage(X, method="ward")


# =============================================================================
# SIDEBAR & MUAT DATA
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 Segmentasi UMKM")
    st.caption("Kabupaten Sukabumi · Data 2024")
    halaman = st.radio(
        "Menu",
        ["🏠 Ringkasan", "🧹 Data & Preprocessing", "🔢 Penentuan Jumlah Cluster",
         "⚖️ Perbandingan Algoritma", "🧩 Profil Segmen", "🗺️ Sebaran Kecamatan",
         "🎯 Cek Segmen UMKM", "📄 Data & Unduhan"],
        label_visibility="collapsed",
    )
    st.divider()
    unggahan = st.file_uploader("Ganti sumber data (opsional)", type=["xlsx"],
                                help="Default: file 'UMKM 2057.xlsx' di folder yang sama dengan app.py. "
                                     "File harus memiliki kolom Asset, Omset, dan Kecamatan.")
    st.caption(
        f"**Metode:** log1p → StandardScaler → K-Means (K={K_FINAL}, random_state={RANDOM_STATE}). "
        f"Pembanding: Hierarchical Ward (K={K_FINAL}) dan DBSCAN (ε={angka(DB_EPS, 2)}, MinPts={DB_MINPTS})."
    )

if unggahan is not None:
    raw = baca_data(unggahan.getvalue())
elif DATA_FILE.exists():
    raw = baca_data(DATA_FILE.read_bytes())
else:
    st.error(
        f"File data tidak ditemukan: **{DATA_FILE.name}**.\n\n"
        "Letakkan file `UMKM 2057.xlsx` satu folder dengan `app.py`, "
        "atau unggah file melalui sidebar."
    )
    st.stop()

if not {"Asset", "Omset"}.issubset(raw.columns):
    st.error("Kolom **Asset** dan **Omset** tidak ditemukan di file. Periksa nama kolom.")
    st.stop()

P = jalankan_pipeline(raw)
df, X = P["df"], P["X"]
N = len(df)

# Tabel evaluasi algoritma
m_km = metrik(X, P["lab_km"])
m_hc = metrik(X, P["lab_hc"])
m_db = metrik(X, P["lab_db"])
eval_algo = pd.DataFrame([
    {"Algoritma": "K-Means", "Parameter": f"K={K_FINAL}", **m_km},
    {"Algoritma": "Hierarchical", "Parameter": f"Ward, K={K_FINAL}", **m_hc},
    {"Algoritma": "DBSCAN", "Parameter": f"ε={angka(DB_EPS, 2)}, MinPts={DB_MINPTS}", **m_db},
])
layak = eval_algo[eval_algo["Lolos Gate"]].reset_index(drop=True)
skor_layak = skor_komposit(layak, tuple(BOBOT_DEFAULT.values())) if len(layak) else layak
terbaik = skor_layak.sort_values("Skor", ascending=False).iloc[0] if len(skor_layak) else None

# Profil segmen
profil = (
    df.groupby(["Cluster", "Segmen_Key", "Segmen"])
    .agg(Jumlah=("Asset", "size"),
         Asset_Rata2=("Asset", "mean"), Asset_Median=("Asset", "median"),
         Asset_Min=("Asset", "min"), Asset_Max=("Asset", "max"),
         Omset_Rata2=("Omset", "mean"), Omset_Median=("Omset", "median"),
         Omset_Min=("Omset", "min"), Omset_Max=("Omset", "max"))
    .reset_index()
    .sort_values("Cluster")
)
profil["Persen"] = profil["Jumlah"] / N * 100


def scatter_segmen(data, judul="", tinggi=520, centroid=True, ekstra=None):
    hover = {c: True for c in ["Nama Usaha", "Kecamatan", "Produksi"] if c in data.columns}
    hover.update({"Asset": ":,.0f", "Omset": ":,.0f", "Segmen": False})
    fig = px.scatter(
        data, x="Asset", y="Omset", color="Segmen", symbol="Segmen",
        color_discrete_map=WARNA_LABEL, symbol_map=SIMBOL_LABEL,
        category_orders={"Segmen": URUTAN_LABEL}, hover_data=hover, opacity=0.75,
        labels={"Asset": "Aset (Rp, skala log)", "Omset": "Omset per tahun (Rp, skala log)"},
        title=judul,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="rgba(255,255,255,0.8)")))
    if centroid:
        c = P["centroid"]
        fig.add_trace(go.Scatter(
            x=c["Asset"], y=c["Omset"], mode="markers", name="Centroid",
            marker=dict(symbol="x", size=16, color="#0b0b0b", line=dict(width=2, color="white")),
            text=c["Segmen"],
            hovertemplate="<b>Centroid</b><br>%{text}<br>Aset: Rp %{x:,.0f}<br>Omset: Rp %{y:,.0f}<extra></extra>",
        ))
    if ekstra is not None:
        fig.add_trace(ekstra)
    sumbu_log(fig)
    return rapikan(fig, tinggi)


def kartu_segmen(key, kolom):
    s = SEGMEN[key]
    baris = profil[profil["Segmen_Key"] == key].iloc[0]
    with kolom:
        with st.container(border=True):
            st.markdown(
                f"<div style='border-left:6px solid {s['warna']};padding-left:10px'>"
                f"<span style='font-size:0.8rem;opacity:0.7'>Cluster {s['no']}</span><br>"
                f"<b>{s['nama']}</b></div>",
                unsafe_allow_html=True,
            )
            st.metric("Jumlah UMKM", angka(baris["Jumlah"]))
            st.caption(f"**{angka(baris['Persen'], 1)}%** dari total UMKM")
            st.caption(f"Median aset **{rp(baris['Asset_Median'])}** · median omset **{rp(baris['Omset_Median'])}**")
            st.caption(s["ciri"])


# =============================================================================
# HALAMAN 1 - RINGKASAN
# =============================================================================
if halaman == "🏠 Ringkasan":
    st.title("Dashboard Segmentasi UMKM Kabupaten Sukabumi")
    st.caption("Perbandingan K-Means, Hierarchical Clustering, dan DBSCAN berdasarkan aset dan omset tahunan UMKM (data Pemerintah Kabupaten Sukabumi, 2024)")

    c = st.columns(5)
    c[0].metric("Data awal", angka(len(raw)))
    c[1].metric("Data dianalisis", angka(N), help=f"{angka(len(raw) - N)} data dikeluarkan saat pembersihan")
    c[2].metric("Jumlah segmen", K_FINAL)
    c[3].metric("Algoritma terbaik", terbaik["Algoritma"] if terbaik is not None else "-")
    c[4].metric("Skor komposit", angka(terbaik["Skor"], 4) if terbaik is not None else "-")

    st.subheader("Empat segmen UMKM")
    cols = st.columns(4)
    for i, key in enumerate(URUTAN_SEGMEN):
        kartu_segmen(key, cols[i])

    kiri, kanan = st.columns([3, 2])
    with kiri:
        tampil(scatter_segmen(df, "Peta aset vs omset per segmen (K-Means, K=4)"))
    with kanan:
        fig = px.bar(
            profil, x="Jumlah", y="Segmen", orientation="h", color="Segmen",
            color_discrete_map=WARNA_LABEL, category_orders={"Segmen": URUTAN_LABEL},
            text=profil.apply(lambda r: f"{angka(r['Jumlah'])} ({angka(r['Persen'], 1)}%)", axis=1),
            title="Jumlah UMKM per segmen",
        )
        fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
        fig.update_layout(yaxis_title=None, xaxis_title="Jumlah UMKM", bargap=0.35)
        tampil(rapikan(fig, 520, legenda=False))

    st.subheader("Temuan utama")
    st.markdown(
        f"""
- **K-Means terpilih** sebagai algoritma terbaik dengan skor komposit **{angka(terbaik['Skor'], 4)}**, unggul atas Hierarchical pada Silhouette, DBI, CHI, dan keseimbangan ukuran cluster pada K yang sama (K=4).
- **DBSCAN** memiliki Silhouette tertinggi ({angka(m_db['Silhouette'], 4)}), tetapi hanya memisahkan {m_db['Terkecil']} UMKM dari {angka(m_db['Terbesar'])} UMKM lainnya sehingga **tidak lolos feasibility gate** (cluster terkecil ≥ 5%).
- Segmen terbesar adalah **{profil.sort_values('Jumlah').iloc[-1]['Segmen']}** ({angka(profil['Persen'].max(), 1)}%).
- Segmen **Usaha Mikro Subsisten** adalah kelompok paling rentan dan menjadi prioritas bantuan dasar.
"""
    )

# =============================================================================
# HALAMAN 2 - DATA & PREPROCESSING
# =============================================================================
elif halaman == "🧹 Data & Preprocessing":
    st.title("Data & Preprocessing")
    st.caption("Tahapan KDD: pembersihan → seleksi variabel → transformasi log1p → standardisasi")

    kiri, kanan = st.columns([2, 3])
    with kiri:
        st.subheader("Alur pembersihan data")
        f = P["funnel"].copy()
        f["Keterangan"] = f["Jumlah"].apply(lambda v: f"{'−' if v < 0 else ''}{angka(abs(v))}")
        tabel(f[["Tahap", "Keterangan"]].rename(columns={"Keterangan": "Jumlah data"}))
        st.info(
            f"**{len(P['df_salah'])} data salah satuan** memiliki aset dan omset hanya ribuan rupiah "
            "(≈400–470× lebih kecil dari kelompok terdekat) dan terkonsentrasi di kecamatan tertentu. "
            "Ini konsisten dengan kesalahan pencatatan satuan (ribu rupiah vs rupiah penuh), sehingga dikeluarkan, bukan dikoreksi."
        )
        with st.expander(f"Lihat {len(P['df_salah'])} data salah satuan"):
            ds = P["df_salah"][["Kecamatan", "Asset", "Omset"]].copy()
            tabel(ds.sort_values("Kecamatan"), height=300)
            tabel(ds["Kecamatan"].value_counts().rename_axis("Kecamatan").reset_index(name="Jumlah"))
    with kanan:
        st.subheader("Mengapa transformasi log?")
        var = st.segmented_control("Variabel", ["Asset", "Omset"], default="Asset") \
            if hasattr(st, "segmented_control") else st.radio("Variabel", ["Asset", "Omset"], horizontal=True)
        var = var or "Asset"
        a, b = st.columns(2)
        with a:
            fig = px.histogram(df, x=var, nbins=40, title=f"{var} asli (Rp)",
                               color_discrete_sequence=["#2a78d6"])
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            fig.update_layout(yaxis_title="Jumlah UMKM", xaxis_title=f"{var} (Rp)", bargap=0.05)
            tampil(rapikan(fig, 340, legenda=False))
        with b:
            d = pd.DataFrame({f"log1p({var})": np.log1p(df[var])})
            fig = px.histogram(d, x=f"log1p({var})", nbins=40, title=f"{var} setelah log1p",
                               color_discrete_sequence=["#1baf7a"])
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            fig.update_layout(yaxis_title="Jumlah UMKM", bargap=0.05)
            tampil(rapikan(fig, 340, legenda=False))
        skew_asli, skew_log = df[var].skew(), np.log1p(df[var]).skew()
        st.caption(f"Skewness {var}: **{angka(skew_asli, 2)}** (asli) → **{angka(skew_log, 2)}** (log1p). "
                   "Distribusi asli sangat menceng ke kanan; tanpa log, K-Means akan 'menghabiskan' cluster untuk outlier.")

    st.subheader("Statistik deskriptif")
    t1, t2, t3 = st.tabs(["Nilai asli (Rp)", "Setelah log1p", "Setelah StandardScaler"])
    with t1:
        s = df[["Asset", "Omset"]].describe().T
        tabel(s.reset_index().rename(columns={"index": "Variabel"}).style.format(
            {c: (lambda v: angka(v)) for c in s.columns}))
    with t2:
        s = pd.DataFrame(P["X_log"], columns=["Asset", "Omset"]).describe().T
        tabel(s.reset_index().rename(columns={"index": "Variabel"}).round(3))
    with t3:
        s = pd.DataFrame(X, columns=["Asset", "Omset"]).describe().T
        tabel(s.reset_index().rename(columns={"index": "Variabel"}).round(3))

    st.subheader("Profil responden")
    a, b, c = st.columns(3)
    with a:
        if "Jenis Kelamin" in df.columns:
            g = df["Jenis Kelamin"].value_counts().reset_index()
            g.columns = ["Jenis Kelamin", "Jumlah"]
            fig = px.bar(g, x="Jenis Kelamin", y="Jumlah", text="Jumlah", title="Jenis kelamin pemilik",
                         color_discrete_sequence=["#2a78d6"])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title=None, bargap=0.45)
            tampil(rapikan(fig, 360, legenda=False))
    with b:
        g = df["Kecamatan"].value_counts().head(10).reset_index()
        g.columns = ["Kecamatan", "Jumlah"]
        fig = px.bar(g, x="Jumlah", y="Kecamatan", orientation="h", text="Jumlah",
                     title="10 kecamatan dengan UMKM terbanyak", color_discrete_sequence=["#2a78d6"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), bargap=0.3)
        tampil(rapikan(fig, 360, legenda=False))
    with c:
        if "Pelatihan yang di ikuti" in df.columns:
            g = df["Pelatihan yang di ikuti"].value_counts().head(8).reset_index()
            g.columns = ["Pelatihan", "Jumlah"]
            g["Pelatihan"] = g["Pelatihan"].str.slice(0, 45) + np.where(g["Pelatihan"].str.len() > 45, "…", "")
            fig = px.bar(g, x="Jumlah", y="Pelatihan", orientation="h", text="Jumlah",
                         title="8 pelatihan paling banyak diikuti", color_discrete_sequence=["#2a78d6"])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis=dict(autorange="reversed", title=None), bargap=0.3)
            tampil(rapikan(fig, 360, legenda=False))

# =============================================================================
# HALAMAN 3 - PENENTUAN K
# =============================================================================
elif halaman == "🔢 Penentuan Jumlah Cluster":
    st.title("Penentuan Jumlah Cluster (K)")
    st.caption("K-Means dijalankan untuk K = 2–10 dan dievaluasi dengan empat indikator")
    ek = evaluasi_k(X)

    def garis(kolom, judul, arah):
        fig = px.line(ek, x="K", y=kolom, markers=True, title=f"{judul}  ·  {arah}",
                      color_discrete_sequence=["#2a78d6"])
        fig.update_traces(line_width=2, marker_size=8)
        pilih = ek[ek["K"] == K_FINAL]
        fig.add_trace(go.Scatter(x=pilih["K"], y=pilih[kolom], mode="markers", name=f"K={K_FINAL} (terpilih)",
                                 marker=dict(size=15, color="#e87ba4", line=dict(width=2, color="white")),
                                 hovertemplate=f"K={K_FINAL}<br>{kolom}: %{{y:.4f}}<extra></extra>"))
        fig.update_layout(xaxis=dict(dtick=1, title="Jumlah cluster (K)"), yaxis_title=None)
        return rapikan(fig, 300, legenda=False)

    a, b = st.columns(2)
    with a:
        tampil(garis("WCSS", "Elbow Method (WCSS)", "cari 'siku'"))
        tampil(garis("DBI", "Davies-Bouldin Index", "lebih rendah lebih baik"))
    with b:
        tampil(garis("Silhouette", "Silhouette Score", "lebih tinggi lebih baik"))
        tampil(garis("CHI", "Calinski-Harabasz Index", "lebih tinggi lebih baik"))

    st.subheader("Skor berbobot per K")
    kiri, kanan = st.columns([3, 2])
    with kiri:
        e = ek.copy()
        e["Warna"] = np.where(e["K"] == K_FINAL, f"K={K_FINAL} (terpilih)", "Kandidat lain")
        fig = px.bar(e, x="K", y="Skor", color="Warna", text=e["Skor"].map(lambda v: angka(v, 3)),
                     color_discrete_map={f"K={K_FINAL} (terpilih)": "#e87ba4", "Kandidat lain": "#2a78d6"},
                     title="Skor komposit (30% Sil + 30% DBI + 20% CHI + 20% Balance)")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis=dict(dtick=1, title="K"), yaxis_title="Skor", bargap=0.35)
        tampil(rapikan(fig, 360))
    with kanan:
        st.markdown(
            f"""
**Mengapa K = 4?**

- **K = 2** mendapat skor tertinggi ({angka(ek.loc[ek.K == 2, 'Skor'].iloc[0], 3)}), tetapi hanya membagi UMKM menjadi dua kelompok besar ({ek.loc[ek.K == 2, 'Ukuran'].iloc[0]}) — terlalu kasar untuk kebijakan yang terdiferensiasi.
- **K = 6** ({angka(ek.loc[ek.K == 6, 'Skor'].iloc[0], 3)}) menghasilkan segmen yang terlalu sempit untuk ditafsirkan dengan hanya dua variabel.
- **K = 4** ({angka(ek.loc[ek.K == 4, 'Skor'].iloc[0], 3)}) menyeimbangkan kelayakan statistik dan kebutuhan segmentasi kebijakan.
"""
        )
    tampilan = ek[["K", "WCSS", "Silhouette", "DBI", "CHI", "Ukuran", "Balance", "Skor"]]
    tabel(tampilan.style.format({"WCSS": "{:.2f}", "Silhouette": "{:.4f}", "DBI": "{:.4f}",
                                 "CHI": "{:.2f}", "Balance": "{:.3f}", "Skor": "{:.4f}"})
          .highlight_between(subset=["K"], left=K_FINAL, right=K_FINAL, color="rgba(232,123,164,0.25)"))

    st.subheader("Pratinjau hasil K-Means untuk K lain")
    k_pilih = st.slider("Pilih K", 2, 10, K_FINAL)
    lab = KMeans(n_clusters=k_pilih, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    d = df[["Asset", "Omset", "Nama Usaha", "Kecamatan"]].copy()
    d["Cluster"] = (lab + 1).astype(str)
    urut = [str(i) for i in range(1, k_pilih + 1)]
    fig = px.scatter(d, x="Asset", y="Omset", color="Cluster", category_orders={"Cluster": urut},
                     hover_data={"Nama Usaha": True, "Kecamatan": True, "Asset": ":,.0f", "Omset": ":,.0f"},
                     opacity=0.75, title=f"K-Means dengan K = {k_pilih}",
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                              "#008300", "#6250d6", "#e34948", "#52514e", "#8a8a85"])
    fig.update_traces(marker=dict(size=7))
    tampil(rapikan(sumbu_log(fig), 480))
    if k_pilih > 3:
        st.caption("Catatan: lebih dari 3 warna pada scatter sulit dibedakan untuk sebagian pembaca; arahkan kursor untuk melihat nomor cluster.")

# =============================================================================
# HALAMAN 4 - PERBANDINGAN ALGORITMA
# =============================================================================
elif halaman == "⚖️ Perbandingan Algoritma":
    st.title("Perbandingan Algoritma Clustering")
    st.caption(f"Semua algoritma memakai ruang fitur yang sama (log1p + StandardScaler). "
               f"Feasibility gate: cluster terkecil ≥ {int(MIN_SHARE*100)}% data (≥ {int(np.ceil(MIN_SHARE*N))} UMKM) dan noise ≤ {int(MAX_NOISE*100)}%.")

    t = eval_algo.copy()
    t["Rentang ukuran"] = t["Terkecil"].map(angka) + " – " + t["Terbesar"].map(angka)
    t["Layak?"] = np.where(t["Lolos Gate"], "✅ Ya", "❌ Tidak")
    tabel(t[["Algoritma", "Parameter", "Cluster", "Noise", "Silhouette", "DBI", "CHI",
             "Rentang ukuran", "Balance", "Layak?"]]
          .style.format({"Silhouette": "{:.4f}", "DBI": "{:.4f}", "CHI": "{:.2f}", "Balance": "{:.3f}"}))

    st.subheader("Metrik per algoritma")
    cols = st.columns(4)
    for col, (kolom, ket) in zip(cols, [("Silhouette", "↑ lebih baik"), ("DBI", "↓ lebih baik"),
                                       ("CHI", "↑ lebih baik"), ("Balance", "↑ lebih seimbang")]):
        with col:
            fig = px.bar(eval_algo, x="Algoritma", y=kolom, color="Algoritma", color_discrete_map=WARNA_ALGO,
                         text=eval_algo[kolom].map(lambda v: angka(v, 2 if kolom == "CHI" else 3)),
                         title=f"{kolom} ({ket})")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title=None, yaxis_title=None, bargap=0.35)
            tampil(rapikan(fig, 320, legenda=False))

    st.subheader("Skor komposit berbobot")
    kiri, kanan = st.columns([2, 3])
    with kiri:
        with st.container(border=True):
            st.markdown("**Atur bobot** (otomatis dinormalisasi agar total = 1)")
            ws = st.slider("Silhouette", 0.0, 1.0, BOBOT_DEFAULT["Silhouette"], 0.05)
            wd = st.slider("DBI (dibalik)", 0.0, 1.0, BOBOT_DEFAULT["DBI"], 0.05)
            wc = st.slider("CHI", 0.0, 1.0, BOBOT_DEFAULT["CHI"], 0.05)
            wb = st.slider("Balance", 0.0, 1.0, BOBOT_DEFAULT["Balance"], 0.05)
            total = ws + wd + wc + wb or 1
            bobot = (ws / total, wd / total, wc / total, wb / total)
            pakai_gate = st.toggle("Terapkan feasibility gate", value=True,
                                   help="Matikan untuk melihat bagaimana DBSCAN 'menang' jika hanya mengandalkan metrik internal.")
    with kanan:
        kandidat = eval_algo[eval_algo["Lolos Gate"]] if pakai_gate else eval_algo
        s = skor_komposit(kandidat.reset_index(drop=True), bobot).sort_values("Skor", ascending=False)
        fig = px.bar(s, x="Skor", y="Algoritma", orientation="h", color="Algoritma",
                     color_discrete_map=WARNA_ALGO, text=s["Skor"].map(lambda v: angka(v, 4)),
                     title="Skor komposit " + ("(hanya kandidat layak)" if pakai_gate else "(tanpa gate)"))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis=dict(range=[0, 1.1], title="Skor"), yaxis=dict(autorange="reversed", title=None),
                          bargap=0.4)
        tampil(rapikan(fig, 300, legenda=False))
        juara = s.iloc[0]
        if pakai_gate:
            st.success(f"**{juara['Algoritma']}** terbaik dengan skor **{angka(juara['Skor'], 4)}**. "
                       f"DBSCAN dikeluarkan karena cluster terkecilnya hanya {m_db['Terkecil']} UMKM "
                       f"({angka(m_db['Terkecil'] / N * 100, 2)}% data).")
        else:
            st.warning(f"Tanpa gate, **{juara['Algoritma']}** memperoleh skor tertinggi — padahal partisinya "
                       "tidak berguna untuk segmentasi. Inilah alasan feasibility gate diperlukan.")

    st.subheader("Uji sensitivitas bobot")
    rows = []
    for nama, w in SKENARIO_BOBOT.items():
        sc = skor_komposit(layak, w).set_index("Algoritma")["Skor"]
        rows.append({"Skenario (Sil/DBI/CHI/Balance)": nama, **{a: sc.get(a, np.nan) for a in layak["Algoritma"]}})
    sens = pd.DataFrame(rows)
    if {"K-Means", "Hierarchical"}.issubset(sens.columns):
        sens["Selisih"] = sens["K-Means"] - sens["Hierarchical"]
    tabel(sens.style.format({c: "{:.4f}" for c in sens.columns if c != "Skenario (Sil/DBI/CHI/Balance)"}))
    st.caption("K-Means tetap unggul di semua skenario, termasuk bobot 100% pada Balance — pemilihan K-Means tidak bergantung pada bobot 30/30/20/20.")

    st.subheader("Detail tiap algoritma")
    tk, th, td = st.tabs(["K-Means", "Hierarchical (Ward)", "DBSCAN"])
    with tk:
        tampil(scatter_segmen(df, f"Segmentasi akhir K-Means (K={K_FINAL})"))
    with th:
        a, b = st.columns(2)
        with a:
            Z = matriks_linkage(X)
            potong = (Z[-K_FINAL, 2] + Z[-(K_FINAL - 1), 2]) / 2
            fig, ax = plt.subplots(figsize=(8, 4.6))
            dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8,
                       show_contracted=True, color_threshold=potong, ax=ax)
            ax.axhline(potong, color="#e34948", ls="--", lw=1.5, label=f"Potongan K={K_FINAL}")
            ax.set_title("Dendrogram Ward linkage (30 kelompok terakhir)", fontweight="bold")
            ax.set_xlabel("Kelompok data (jumlah anggota)")
            ax.set_ylabel("Jarak Euclidean")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with b:
            d = df[["Asset", "Omset", "Nama Usaha", "Kecamatan"]].copy()
            d["Cluster HC"] = (P["lab_hc"] + 1).astype(str)
            fig = px.scatter(d, x="Asset", y="Omset", color="Cluster HC",
                             category_orders={"Cluster HC": ["1", "2", "3", "4"]},
                             color_discrete_sequence=["#2a78d6", "#eda100", "#1baf7a", "#e87ba4"],
                             hover_data={"Nama Usaha": True, "Kecamatan": True, "Asset": ":,.0f", "Omset": ":,.0f"},
                             opacity=0.75, title=f"Hierarchical Ward (K={K_FINAL})")
            fig.update_traces(marker=dict(size=7))
            tampil(rapikan(sumbu_log(fig), 460))
        ukuran = pd.Series(P["lab_hc"] + 1).value_counts().sort_index()
        st.caption("Ukuran cluster Hierarchical: " + ", ".join(f"C{i}: {v}" for i, v in ukuran.items()))
    with td:
        a, b = st.columns(2)
        with a:
            mp = st.selectbox("MinPts untuk K-distance plot", [5, 10, 15, 20], index=0)
            kd = k_distance(X, mp)
            fig = px.line(x=np.arange(len(kd)), y=kd, title=f"K-distance plot (MinPts = {mp})",
                          labels={"x": "Indeks data (terurut)", "y": f"Jarak ke tetangga ke-{mp}"},
                          color_discrete_sequence=["#2a78d6"])
            fig.add_hline(y=DB_EPS, line_dash="dash", line_color="#e34948",
                          annotation_text=f"ε = {angka(DB_EPS, 2)}", annotation_position="top left")
            tampil(rapikan(fig, 380, legenda=False))
        with b:
            d = df[["Asset", "Omset", "Nama Usaha", "Kecamatan"]].copy()
            d["Label DBSCAN"] = np.where(P["lab_db"] == -1, "Noise", "Cluster " + (P["lab_db"] + 1).astype(str))
            fig = px.scatter(d, x="Asset", y="Omset", color="Label DBSCAN",
                             color_discrete_map={"Cluster 1": "#2a78d6", "Cluster 2": "#eda100", "Noise": "#8a8a85"},
                             hover_data={"Nama Usaha": True, "Kecamatan": True, "Asset": ":,.0f", "Omset": ":,.0f"},
                             opacity=0.75, title=f"DBSCAN (ε={angka(DB_EPS, 2)}, MinPts={DB_MINPTS})")
            fig.update_traces(marker=dict(size=7))
            tampil(rapikan(sumbu_log(fig), 380))
        st.caption("Ukuran: " + ", ".join(f"{k}: {v}" for k, v in d["Label DBSCAN"].value_counts().items()))
        with st.expander("Hasil grid search DBSCAN (10 konfigurasi terbaik)"):
            g = grid_dbscan(X)
            if len(g):
                g = g.sort_values("Skor", ascending=False).head(10)
                tabel(g[["eps", "MinPts", "Cluster", "Noise", "Silhouette", "DBI", "CHI", "Terkecil", "Terbesar", "Skor"]]
                      .style.format({"eps": "{:.2f}", "Silhouette": "{:.4f}", "DBI": "{:.4f}", "CHI": "{:.2f}", "Skor": "{:.4f}"}))
                st.caption("Hanya konfigurasi dengan 2–10 cluster dan noise ≤ 10%. Semua konfigurasi tetap didominasi satu cluster besar.")

# =============================================================================
# HALAMAN 5 - PROFIL SEGMEN
# =============================================================================
elif halaman == "🧩 Profil Segmen":
    st.title("Profil Segmen UMKM")
    kec = st.multiselect("Filter kecamatan (kosongkan = semua)", sorted(df["Kecamatan"].unique()))
    data = df[df["Kecamatan"].isin(kec)] if kec else df
    if data.empty:
        st.warning("Tidak ada data untuk filter ini.")
        st.stop()
    st.caption(f"Menampilkan **{angka(len(data))}** dari {angka(N)} UMKM")

    tampil(scatter_segmen(data, "Sebaran aset dan omset per segmen", centroid=not kec))

    pf = (data.groupby(["Cluster", "Segmen"])
          .agg(Jumlah=("Asset", "size"), **{"Rata-rata aset": ("Asset", "mean"), "Median aset": ("Asset", "median"),
                                            "Rata-rata omset": ("Omset", "mean"), "Median omset": ("Omset", "median"),
                                            "Omset minimum": ("Omset", "min"), "Omset maksimum": ("Omset", "max")})
          .reset_index().sort_values("Cluster"))
    pf.insert(3, "Persen", pf["Jumlah"] / len(data) * 100)
    st.subheader("Tabel profil")
    tabel(pf.drop(columns="Cluster").style.format(
        {"Persen": "{:.2f}%", **{c: rp_penuh for c in pf.columns if c not in ["Cluster", "Segmen", "Jumlah", "Persen"]}}))

    a, b = st.columns(2)
    for kol, var in zip([a, b], ["Asset", "Omset"]):
        with kol:
            fig = px.box(data, x="Segmen", y=var, color="Segmen", color_discrete_map=WARNA_LABEL,
                         category_orders={"Segmen": URUTAN_LABEL}, points=False,
                         title=f"Distribusi {var.lower()} per segmen (skala log)")
            fig.update_layout(xaxis=dict(title=None, tickvals=URUTAN_LABEL,
                                         ticktext=[f"C{SEGMEN[k]['no']}" for k in URUTAN_SEGMEN]))
            sumbu_log(fig, x=False)
            tampil(rapikan(fig, 380))

    st.subheader("Detail & rekomendasi kebijakan")
    pilih = st.selectbox("Pilih segmen", URUTAN_SEGMEN, format_func=lambda k: LABEL_SEGMEN[k])
    s = SEGMEN[pilih]
    sub = data[data["Segmen_Key"] == pilih]
    a, b, c = st.columns([2, 2, 3])
    with a:
        st.markdown(f"<div style='border-left:6px solid {s['warna']};padding-left:10px'>"
                    f"<b>{s['nama']}</b><br><span style='opacity:0.7'>{s['en']}</span></div>",
                    unsafe_allow_html=True)
        st.write(s["ciri"])
        st.markdown("**Rekomendasi intervensi:**")
        for r in s["rekomendasi"]:
            st.markdown(f"- {r}")
    with b:
        top = sub["Kecamatan"].value_counts().head(8).reset_index()
        top.columns = ["Kecamatan", "Jumlah"]
        fig = px.bar(top, x="Jumlah", y="Kecamatan", orientation="h", text="Jumlah",
                     color_discrete_sequence=[s["warna"]], title="Kecamatan terbanyak")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), bargap=0.3)
        tampil(rapikan(fig, 340, legenda=False))
    with c:
        kol = [k for k in ["Nama Usaha", "Kecamatan", "Produksi", "Asset", "Omset"] if k in sub.columns]
        st.markdown(f"**Contoh anggota** ({angka(len(sub))} UMKM)")
        tabel(sub[kol].sort_values("Omset", ascending=False).head(50)
              .style.format({"Asset": rp_penuh, "Omset": rp_penuh}), height=300)

# =============================================================================
# HALAMAN 6 - SEBARAN KECAMATAN
# =============================================================================
elif halaman == "🗺️ Sebaran Kecamatan":
    st.title("Sebaran Segmen per Kecamatan")
    ct = pd.crosstab(df["Kecamatan"], df["Segmen"]).reindex(columns=URUTAN_LABEL, fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    ct = ct.sort_values("Total", ascending=False)

    a, b, c = st.columns([2, 2, 3])
    n_top = a.slider("Jumlah kecamatan ditampilkan", 5, len(ct), min(20, len(ct)))
    mode = b.radio("Tampilan", ["Jumlah", "Proporsi (%)"], horizontal=True)
    urut = c.selectbox("Urutkan berdasarkan", ["Total UMKM"] + URUTAN_LABEL)

    d = ct.head(n_top) if urut == "Total UMKM" else \
        ct.assign(_p=ct[urut] / ct["Total"]).query("Total >= 5").sort_values("_p", ascending=False).head(n_top).drop(columns="_p")
    if urut != "Total UMKM":
        st.caption("Pengurutan berdasarkan proporsi segmen; hanya kecamatan dengan ≥ 5 UMKM.")
    long = d.drop(columns="Total").reset_index().melt(id_vars="Kecamatan", var_name="Segmen", value_name="Jumlah")
    if mode == "Proporsi (%)":
        long["Nilai"] = long["Jumlah"] / long["Kecamatan"].map(d["Total"]) * 100
    else:
        long["Nilai"] = long["Jumlah"]
    fig = px.bar(long, x="Nilai", y="Kecamatan", color="Segmen", orientation="h",
                 color_discrete_map=WARNA_LABEL, category_orders={"Segmen": URUTAN_LABEL, "Kecamatan": list(d.index)},
                 custom_data=["Segmen", "Jumlah"],
                 title=f"Komposisi segmen pada {len(d)} kecamatan")
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Jumlah: %{customdata[1]}<br>"
                                    + ("Proporsi: %{x:.1f}%" if mode != "Jumlah" else "") + "<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_title=None, bargap=0.25,
                      xaxis_title="Proporsi (%)" if mode != "Jumlah" else "Jumlah UMKM")
    tampil(rapikan(fig, max(420, 26 * len(d) + 120)))

    st.subheader("Komposisi satu kecamatan")
    kc = st.selectbox("Pilih kecamatan", list(ct.index))
    baris = ct.loc[kc]
    cols = st.columns(5)
    cols[0].metric("Total UMKM", angka(baris["Total"]))
    for i, lab in enumerate(URUTAN_LABEL):
        cols[i + 1].metric(lab.split(" · ")[1], angka(baris[lab]))
        cols[i + 1].caption(f"{angka(baris[lab] / baris['Total'] * 100, 1)}% dari kecamatan ini")

    with st.expander("Tabel lengkap kecamatan × segmen"):
        tabel(ct.reset_index())
        st.download_button("⬇️ Unduh CSV", ct.to_csv().encode("utf-8"), "sebaran_kecamatan_segmen.csv", "text/csv")

# =============================================================================
# HALAMAN 7 - CEK SEGMEN UMKM BARU
# =============================================================================
elif halaman == "🎯 Cek Segmen UMKM":
    st.title("Cek Segmen UMKM")
    st.caption("Masukkan aset dan omset tahunan sebuah UMKM; sistem memprediksi segmennya memakai model K-Means yang sama.")

    def prediksi(asset, omset):
        z = P["scaler"].transform(np.log1p(np.column_stack([asset, omset])))
        idx = P["kmeans"].predict(z)
        return [P["peta"][i] for i in idx]

    t1, t2 = st.tabs(["Satu UMKM", "Banyak UMKM (unggah Excel)"])
    with t1:
        a, b = st.columns([2, 3])
        with a:
            with st.form("form_cek"):
                nama = st.text_input("Nama usaha (opsional)")
                asset = st.number_input("Total aset (Rp)", min_value=0, value=5_000_000, step=500_000, format="%d")
                omset = st.number_input("Omset per tahun (Rp)", min_value=0, value=24_000_000, step=1_000_000, format="%d")
                kirim = st.form_submit_button("Prediksi segmen", type="primary")
            st.caption(f"Pratinjau input: aset **{rp(asset)}**, omset **{rp(omset)}**")
        with b:
            if kirim:
                if asset <= 0 or omset <= 0:
                    st.error("Aset dan omset harus lebih dari 0.")
                else:
                    if asset <= UNIT_ERR_ASSET and omset <= UNIT_ERR_OMSET:
                        st.warning("Nilai sangat kecil — mungkin tercatat dalam ribuan rupiah. "
                                   "Data seperti ini dikeluarkan saat pelatihan model. Periksa kembali satuannya.")
                    key = prediksi([asset], [omset])[0]
                    s = SEGMEN[key]
                    st.markdown(
                        f"<div style='border-left:8px solid {s['warna']};padding:8px 14px;"
                        f"background:rgba(127,127,127,0.08);border-radius:6px'>"
                        f"<span style='opacity:0.7'>{nama or 'UMKM ini'} termasuk</span><br>"
                        f"<span style='font-size:1.5rem;font-weight:700'>Cluster {s['no']} · {s['nama']}</span><br>"
                        f"{s['ciri']}</div>", unsafe_allow_html=True)
                    st.markdown("**Rekomendasi intervensi:**")
                    for r in s["rekomendasi"]:
                        st.markdown(f"- {r}")
            else:
                st.info("Isi formulir di kiri lalu klik **Prediksi segmen**.")
        if kirim and asset > 0 and omset > 0:
            titik = go.Scatter(x=[asset], y=[omset], mode="markers+text", name="UMKM yang dicek",
                               text=[nama or "UMKM yang dicek"], textposition="top center",
                               marker=dict(symbol="star", size=24, color="#e34948", line=dict(width=2, color="white")),
                               hovertemplate="Aset: Rp %{x:,.0f}<br>Omset: Rp %{y:,.0f}<extra></extra>")
            tampil(scatter_segmen(df, "Posisi UMKM terhadap seluruh data", ekstra=titik))

    with t2:
        st.markdown("Unggah file Excel dengan kolom **Asset** dan **Omset** (kolom lain boleh ada).")
        contoh = pd.DataFrame({"Nama Usaha": ["Contoh A", "Contoh B"], "Asset": [1_000_000, 30_000_000],
                               "Omset": [2_000_000, 150_000_000]})
        st.download_button("⬇️ Unduh templat CSV", contoh.to_csv(index=False).encode("utf-8"), "templat_cek_segmen.csv")
        f = st.file_uploader("File UMKM baru", type=["xlsx", "csv"], key="batch")
        if f is not None:
            baru = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            baru.columns = baru.columns.str.strip()
            if not {"Asset", "Omset"}.issubset(baru.columns):
                st.error("Kolom Asset dan Omset wajib ada.")
            else:
                baru["Asset"] = pd.to_numeric(baru["Asset"], errors="coerce")
                baru["Omset"] = pd.to_numeric(baru["Omset"], errors="coerce")
                ok = (baru["Asset"] > 0) & (baru["Omset"] > 0)
                baru["Segmen"] = "Tidak valid (Asset/Omset kosong atau 0)"
                if ok.any():
                    keys = prediksi(baru.loc[ok, "Asset"].values, baru.loc[ok, "Omset"].values)
                    baru.loc[ok, "Segmen"] = [LABEL_SEGMEN[k] for k in keys]
                tabel(baru)
                st.download_button("⬇️ Unduh hasil prediksi (CSV)", baru.to_csv(index=False).encode("utf-8"),
                                   "hasil_prediksi_segmen.csv", "text/csv")

# =============================================================================
# HALAMAN 8 - DATA & UNDUHAN
# =============================================================================
elif halaman == "📄 Data & Unduhan":
    st.title("Data Hasil Segmentasi")
    a, b = st.columns([2, 3])
    cari = a.text_input("Cari nama usaha / pemilik / produk")
    seg = b.multiselect("Segmen", URUTAN_LABEL, default=URUTAN_LABEL)
    kec = st.multiselect("Kecamatan", sorted(df["Kecamatan"].unique()))

    data = df[df["Segmen"].isin(seg)]
    if kec:
        data = data[data["Kecamatan"].isin(kec)]
    if cari:
        kolom_cari = [c for c in ["Nama Usaha", "Nama Lengkap", "Produksi"] if c in data.columns]
        m = np.zeros(len(data), dtype=bool)
        for c in kolom_cari:
            m |= data[c].astype(str).str.contains(cari, case=False, na=False).values
        data = data[m]

    # Data pribadi (No HP, alamat, tanggal lahir) tidak ditampilkan
    kolom = [c for c in ["Nama Lengkap", "Nama Usaha", "Jenis Kelamin", "Kecamatan", "Produksi",
                         "Pelatihan yang di ikuti", "Asset", "Omset", "Cluster", "Segmen"] if c in data.columns]
    st.caption(f"{angka(len(data))} baris · kolom No HP, alamat, dan tanggal lahir disembunyikan")
    tabel(data[kolom].style.format({"Asset": rp_penuh, "Omset": rp_penuh}), height=480)

    st.subheader("Unduh")
    import io

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df[kolom + ["Cluster_HC", "Cluster_DBSCAN"]].to_excel(w, sheet_name="Hasil Cluster", index=False)
        profil.drop(columns="Segmen_Key").to_excel(w, sheet_name="Profil Cluster", index=False)
        eval_algo.to_excel(w, sheet_name="Evaluasi Algoritma", index=False)
        evaluasi_k(X).drop(columns=["Sil_norm", "DBI_norm", "CHI_norm"]).to_excel(w, sheet_name="Evaluasi K", index=False)
        P["df_salah"][["Kecamatan", "Asset", "Omset"]].to_excel(w, sheet_name="Data Salah Satuan", index=False)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Excel lengkap (5 sheet)", buf.getvalue(), "Hasil_Segmentasi_UMKM_Sukabumi.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇️ CSV data yang difilter", data[kolom].to_csv(index=False).encode("utf-8"),
                       "data_umkm_filter.csv", "text/csv")
