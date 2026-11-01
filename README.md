# Jambi Wind Correction MLOps

Proyek ini dibuat untuk mengeksplorasi bagaimana **machine learning bisa digunakan sebagai lapisan koreksi** terhadap prediksi kecepatan angin jangka pendek dari ECMWF IFS. Lokasi studi yang dipakai adalah Kota Jambi, dengan horizon prakiraan **+6, +12, +18, dan +24 jam**.

Model tidak menggantikan ECMWF IFS. Yang dipelajari adalah selisih antara prakiraan dan data rujukan (*residual*), lalu residual prediksi ditambahkan ke raw forecast untuk menghasilkan nilai terkoreksi.

```text
ECMWF raw forecast
      +
Predicted residual (ML)
      ↓
Corrected wind speed = max(0, raw forecast + predicted residual)
```

> **Catatan data:** Reference yang dipakai pada eksperimen saat ini adalah ECMWF IFS Analysis. Data tersebut berbasis model dan asimilasi, **bukan observasi stasiun cuaca independen**. Proyek masih berstatus akademik/proof-of-concept, bukan layanan prakiraan operasional.

## Perkembangan proyek

Pengerjaan dilakukan bertahap mengikuti lembar kerja mata kuliah MLOps. Penjelasan teknis dan hasil masing-masing tahap ada di folder [`docs/`](docs/README.md).

| LK | Yang dikerjakan | Dokumen |
|---|---|---|
| 01 | Studi masalah, validasi sumber data, dan rencana MLOps | [LK01](docs/LK01-inisiasi-proyek.md) |
| 02 | Struktur GitHub, Codespaces, dan environment | [LK02](docs/LK02-infrastruktur-pengembangan.md) |
| 03 | Desain pipeline dan technical preview forecast–reference | [LK03](docs/LK03-perancangan-pipeline-data.md) |
| 04 | Ingestion, preprocessing, serta GitHub Actions | [LK04](docs/LK04-pipeline-data-dinamis.md) |
| 05 | Versioning dataset dengan DVC | [LK05](docs/LK05-dataset-versioning.md) |
| 06 | Training dan perbandingan tiga Random Forest di MLflow | [LK06](docs/LK06-experiment-tracking.md) |
| 07 | Registry V1/V2, simulasi lifecycle, dan inferensi | [LK07](docs/LK07-model-registry.md) |

## Bagian yang sudah diimplementasikan

- **Pipeline data:** ingestion ECMWF IFS, fallback run, validasi, preprocessing, fitur siklik, dan GitHub Actions terjadwal.
- **Dataset:** riwayat data dan dataset training dikelola dengan metadata DVC; objek data masih bergantung pada cache/backup lokal.
- **Training:** tiga konfigurasi Random Forest dilatih dan dibandingkan menggunakan split waktu, lalu dicatat di MLflow.
- **Model Registry:** versi 1 dan 2 telah dikelola, dengan `champion` pada V1 dan `challenger` pada V2; inferensi sintetis melalui alias sudah diuji.

Beberapa komponen seperti serving API, monitoring drift, dan continuous training otomatis **masih rencana**, bukan fitur yang sudah berjalan end-to-end.

## Ringkasan hasil model terakhir

| Model | MAE validasi (m/s) | RMSE validasi (m/s) |
|---|---:|---:|
| Raw ECMWF (baseline) | 0,6183 | **0,7345** |
| RF-01 / Registry V1 | **0,6152** | 0,8084 |
| RF-02 / Registry V2 | 0,6512 | 0,8413 |
| RF-03 | 0,6960 | 0,8834 |

RF-01 dipilih sebagai champion **untuk simulasi LK07** karena mengungguli RF-02 pada dua metrik. Akan tetapi, RMSE RF-01 masih lebih tinggi daripada raw ECMWF. Dengan demikian, keputusan itu **bukan** klaim bahwa model sudah siap digunakan di produksi nyata.

## Menjalankan bagian proyek

Lingkungan utama pengembangan menggunakan **Python 3.11** melalui GitHub Codespaces. Dari root repo:

```bash
python -m pip install -r requirements.txt

# Pengambilan dan pengolahan forecast
python src/ingest_data.py
python src/preprocess.py

# Memeriksa Registry yang sudah disiapkan di MLflow
python src/lk07_registry.py verify

# Uji inferensi melalui alias champion
python src/lk07_inference.py --alias champion
```

Perintah Registry/inferensi memerlukan **MLflow tracking server dan artefak model yang sesuai**. Dataset yang dilacak DVC juga harus tersedia di cache/remote untuk training ulang; file pointer `.dvc` saja tidak berisi CSV. Lihat masing-masing dokumen LK sebelum menjalankan ulang eksperimen.

## Teknologi

Python, pandas, NumPy, scikit-learn, Open-Meteo API, GitHub Actions, DVC, MLflow, serta GitHub Codespaces. FastAPI, Docker, Airflow, dan monitoring operasional merupakan arah pengembangan berikutnya.

## Lisensi

Repository menggunakan [MIT License](LICENSE).
