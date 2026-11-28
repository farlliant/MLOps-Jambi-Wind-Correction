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
| 08 | Code-triggered testing, retraining, evaluation, dan Model Registry | [LK08](docs/LK08-code-as-trigger.md) |
| 09 | Docker Compose, FastAPI inference, PostgreSQL, dan persistensi | [LK09](docs/LK09-docker-compose.md) |
| 10 | MLflow Models REST Serving dan tiga replika Docker Compose | [LK10](docs/LK10-mlflow-model-serving.md) |
| 11 | Prometheus, Grafana, monitoring tiga replika, dan simulasi drift | [LK11](docs/LK11-monitoring.md) |

## Bagian yang sudah diimplementasikan

- **Pipeline data:** ingestion ECMWF IFS, fallback run, validasi, preprocessing, fitur siklik, dan GitHub Actions terjadwal.
- **Dataset:** riwayat data dan dataset training dikelola dengan metadata DVC; objek data masih bergantung pada cache/backup lokal.
- **Training:** tiga konfigurasi Random Forest dilatih dan dibandingkan menggunakan split waktu, lalu dicatat di MLflow.
- **Model Registry:** versi 1 dan 2 telah dikelola, dengan `champion` pada V1 dan `challenger` pada V2; inferensi sintetis melalui alias sudah diuji.

Serving API FastAPI dan PostgreSQL telah diuji melalui Docker Compose pada LK09. LK10 menambahkan REST API MLflow Models serta tiga replika serving yang diuji secara independen. LK11 menambahkan Prometheus, Grafana, dan deteksi perubahan mean input sintetis pada tiga replika. Monitoring akurasi berbasis reference aktual dan continuous training otomatis masih merupakan rencana.

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

Python, pandas, NumPy, scikit-learn, Open-Meteo API, GitHub Actions, DVC, MLflow, GitHub Codespaces, FastAPI, Docker Compose, dan PostgreSQL. Airflow serta monitoring operasional masih merupakan arah pengembangan.

## Menjalankan LK09 melalui Docker Compose

Isi `.env` dari template `.env.example` dengan password PostgreSQL, access token DagsHub, serta variabel password Grafana yang disyaratkan konfigurasi Compose LK11 (jangan commit `.env`). Dari root repository:

```bash
docker compose up -d --build --wait --wait-timeout 240 api db
docker compose ps
curl -fsS http://127.0.0.1:8000/health
```

Perintah tersebut menjalankan FastAPI dan PostgreSQL sebagai dua service di custom bridge network, dengan named volume untuk riwayat prediksi. Lihat [dokumentasi LK09](docs/LK09-docker-compose.md) untuk konfigurasi, pengujian inferensi, dan evidence persistensi.

## Menjalankan LK10: MLflow Models Serving dan Scaling

LK10 melayani model melalui endpoint native MLflow `/invocations` dan menambahkan tiga replika `model-serving` pada Compose tanpa menghapus layanan LK09. Pastikan `.env` dan akses model DagsHub tersedia, serta environment Python 3.11 dan Docker aktif.

```bash
# Dari root repository; model sumber Version 2 pada Staging
source .venv/bin/activate
python -m src.lk10_model_ops status
python -m src.lk10_model_ops download

# Buat image model (sekali per perubahan artefak)
mlflow models build-docker --model-uri .lk10/model --name jambi-wind-lk10-mlflow --env-manager virtualenv

# Definisi service model-serving sudah berada di docker-compose.yaml
# Jika service belum ditambahkan pada clone lama, jalankan:
# python lk10_apply_compose.py

docker compose up -d --wait --wait-timeout 600 model-serving
docker compose ps model-serving
python -m src.lk10_test_serving --all-replicas
```

Cari port host masing-masing replika (dinamis):

```bash
docker compose port --index 1 model-serving 8080
docker compose port --index 2 model-serving 8080
docker compose port --index 3 model-serving 8080
```

Contoh mengganti jumlah replika dari tiga ke dua, lalu mengembalikan menjadi tiga:

```bash
docker compose up -d --no-deps --scale model-serving=2 model-serving
docker compose up -d --no-deps --scale model-serving=3 model-serving
```

Registry Production demonstrasi LK10 adalah salinan model sumber Version 2 yang ditaruh pada model terpisah `Jambi-Wind-Residual-RandomForest-LK10-Demo` Version 1; model asli tidak dipromosikan dan LK09 tetap memakai Staging. Untuk serving demo Production pada port 5002: `python -m src.lk10_registry_demo serve`, lalu `python -m src.lk10_test_serving --url http://127.0.0.1:5002`. Ini simulasi akademik, bukan persetujuan meteorologi operasional.

Lihat [dokumentasi LK10](docs/LK10-mlflow-model-serving.md) serta [evidence LK10](docs/evidence/lk10/).

## Menjalankan LK11: Prometheus dan Grafana

LK11 menambahkan monitoring pada tiga replika MLflow. Port internal 8080 tetap melayani API native LK10; request yang dipantau harus menggunakan proxy internal 8081. Prometheus melakukan DNS discovery dan scraping `/metrics` setiap 5 detik. Grafana memuat dashboard 12 panel secara otomatis.

Pastikan `.env` lengkap (termasuk `LK11_GRAFANA_PASSWORD`), Python environment aktif, dan image `jambi-wind-lk10-mlflow:latest` sudah dibangun sesuai LK10. Pada konfigurasi final, tidak perlu menjalankan ulang `lk11_apply_compose.py`.

```bash
source .venv/bin/activate
docker compose config --quiet
docker compose up -d --wait --wait-timeout 600 model-serving prometheus grafana
python -m src.lk11_check_monitoring --expect targets
```

Gunakan port 3000 untuk Grafana (user `admin`, password lokal saat inisialisasi) dan port 9090 untuk Prometheus (`/targets`, `/alerts`). Di Codespaces gunakan forwarded address Private. Dashboard berada di Dashboards → MLOps → LK11 | Jambi Wind Model Observability.

```bash
python -m src.lk11_workload --phase baseline --seconds 75
python -m src.lk11_check_monitoring --expect baseline
python -m src.lk11_workload --phase shift --seconds 75
python -m src.lk11_check_monitoring --expect shift
python -m src.lk11_workload --phase recovery --seconds 75
python -m src.lk11_check_monitoring --expect recovery
```

Hasil pengujian 28 November 2026: 675 request sukses, 0 gagal, tiga target UP, drift flags 0 → 1 → 0 pada seluruh replika dan alert drift firing → inactive. Deteksi ini menggunakan perubahan mean input sintetis; bukan bukti penurunan akurasi model atau kapasitas produksi. Lihat [dokumentasi LK11](docs/LK11-monitoring.md) dan [evidence](docs/evidence/lk11/).

## Lisensi

Repository menggunakan [MIT License](LICENSE).
