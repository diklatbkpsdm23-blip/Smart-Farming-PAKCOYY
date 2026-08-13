# 📸 Deteksi Kesiapan Panen dari Foto Daun Pakcoy — Model Terlatih (3 Kelas)

Model klasifikasi untuk halaman **"Deteksi Siap Panen (CNN 2D)"** di
Smart Farming Pakcoy, dilatih dari foto asli kebun kamu — sekarang **3
kelas**: `masa_pertumbuhan`, `mendekati_panen`, `panen`. Sudah terlatih dan
siap pakai — tinggal `streamlit run app.py`, tidak perlu langkah tambahan.

## Apa yang sebenarnya terjadi (biar tetap transparan)

Sama seperti sebelumnya: sandbox saya tidak punya TensorFlow dan tidak ada
internet untuk instal, jadi saya tidak bisa melatih CNN Keras (`.h5`) asli
di sini. Model bawaan ini adalah **classifier scikit-learn** (SVM) dari
fitur warna/tekstur daun (histogram Hue/Saturation, cakupan area hijau,
kekasaran tekstur) — dilatih & divalidasi sungguhan dari foto asli kamu,
bukan aturan hardcode.

`app.py` otomatis pakai `pakcoy_cnn2d_sklearn.pkl`. Kalau nanti kamu
jalankan `train_cnn2d_pakcoy_colab.ipynb` di Google Colab (yang sudah ada
TensorFlow), `app.py` otomatis pindah pakai `.h5` CNN itu — prioritasnya
lebih tinggi, tidak perlu ubah kode.

## Data & hasil evaluasi

| Kelas | Sumber | Jumlah foto asli |
|---|---|---|
| `masa_pertumbuhan` | Masa_Pertumbuhan.rar | 102 |
| `mendekati_panen` | mendekati_panen.rar | 19 |
| `panen` | panen.rar | 7 |

Model dipilih otomatis dari 3 kandidat lewat 5-fold cross-validation.
Yang menang lagi: **SVM (RBF kernel)**.

- **F1-macro cross-validation: 0.868** (rata-rata 5 fold, std 0.113)
- **Akurasi holdout 20%: 88%**

**Realistis, harus digarisbawahi:** kelas `panen` cuma punya **7 foto**.
Itu jauh di bawah jumlah yang layak untuk classifier yang bisa diandalkan
— skor di atas untuk kelas ini praktis dihitung dari 1 foto per fold
cross-validation, jadi sangat rentan berubah drastis begitu ada foto baru
yang beda kondisi (pencahayaan, sudut, latar). Anggap performa kelas
`panen` saat ini sebagai **prototipe kasar**, bukan angka yang bisa
dipercaya penuh. Tambah minimal 20–30 foto lagi untuk kelas ini secepatnya
kalau mau dipakai serius.

## Isi paket ini

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit, halaman CNN 2D otomatis pakai model bawaan |
| `pakcoy_cnn2d_sklearn.pkl` | **Model terlatih 3 kelas** (SVM + fitur warna/tekstur) |
| `class_order.json` | Urutan label kelas, dibaca otomatis oleh `app.py` |
| `pakcoy_features.py` | Modul ekstraksi fitur, dipakai bersama training & app |
| `train_sklearn.py` | Skrip training (scikit-learn) — jalankan ulang kalau tambah foto |
| `train_cnn2d_pakcoy_colab.ipynb` | Notebook Colab untuk CNN Keras asli (opsional, TensorFlow) |
| `train_cnn2d_pakcoy.py` | Versi skrip lokal dari notebook di atas |
| `requirements.txt` | Dependency aplikasi |
| `dataset_pakcoy_foto.zip` | Foto asli kamu (3 kelas), sudah dibersihkan & disusun |

## Cara jalan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka halaman **"📸 Deteksi Siap Panen (CNN 2D)"** — notifikasi hijau
menandakan model 3-kelas bawaan aktif.

## Kalau mau tambah foto lagi (terutama kelas `panen`)

1. Ekstrak `dataset_pakcoy_foto.zip`, tambah foto baru ke folder kelas
   yang sesuai (`dataset/panen/`, dst).
2. `python train_sklearn.py` — otomatis retrain & timpa `pakcoy_cnn2d_sklearn.pkl`.

Bagian lain `app.py` (Dashboard, Eksplorasi Data, Model AI sensor,
Monitoring Realtime) tidak diubah — hanya halaman CNN 2D yang disentuh.
