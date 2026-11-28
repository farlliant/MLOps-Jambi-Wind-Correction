# LK11 — Observabilitas Model dengan Prometheus dan Grafana

## Tujuan dan arsitektur

LK11 melanjutkan tiga replika native MLflow Models Serving LK10 dengan pemantauan operasional serta deteksi perubahan distribusi input. Service FastAPI dan PostgreSQL LK09, network `wind-network`, serta volume `postgres_data` dipertahankan. Model tidak dilatih ulang atau dipromosikan pada tahap ini.

Setiap container `model-serving` menjalankan command native MLflow dari image LK10 sebagai child process dan pembungkus HTTP `monitoring/lk11_proxy.py` sebagai entrypoint. Proxy meneruskan `/ping` dan `/invocations` ke native scoring server pada port 8080, serta menyediakan `/metrics` pada port 8081. Port native tetap tersedia untuk kompatibilitas LK10, tetapi request yang melewati port 8080 langsung tidak dihitung oleh monitoring LK11. Workload LK11 menggunakan port proxy 8081.

Prometheus v3.5.0 menemukan alamat IP setiap replika melalui DNS service discovery untuk `model-serving`, mengambil metrik setiap 5 detik, dan mengevaluasi alert setiap 5 detik. Grafana v12.1.0 menggunakan datasource Prometheus dan dashboard yang diprovisi otomatis melalui file konfigurasi. Volume `prometheus_data` dan `grafana_data` mempertahankan data monitoring di luar filesystem sementara container; retensi Prometheus disetel 2 hari.

```text
Workload sintetis (3 klien bersamaan)
  -> proxy 8081 masing-masing replika -> MLflow 8080
  -> /metrics per replika -> Prometheus -> Grafana
```

## Reproduksi

Gunakan Python 3.11, virtual environment proyek, Docker dan Docker Compose. Dari root repository, siapkan `.env` dari `.env.example` dengan kredensial PostgreSQL, akses DagsHub dan password Grafana yang kuat. Jangan commit `.env` atau artefak `.lk10/`. Apabila image LK10 belum tersedia, ikuti [panduan LK10](LK10-mlflow-model-serving.md) untuk mengunduh model dan membangun image `jambi-wind-lk10-mlflow:latest` terlebih dahulu. Alias Registry dapat berubah, sehingga untuk mereproduksi angka historis diperlukan artefak model sumber LK10 yang sama (Version 2, run `007309b8db124732804cbdea3acda948`).

Konfigurasi Compose LK11 sudah memuat wrapper dan kedua layanan monitoring. `lk11_apply_compose.py` hanya diperlukan ketika menerapkan patch pada konfigurasi LK10 yang belum memiliki monitoring, dengan image LK10 sudah tersedia. Skrip membaca command asli image dan membuat password Grafana lokal apabila belum ada; tidak perlu menjalankannya ulang pada konfigurasi final.

```bash
source .venv/bin/activate
docker compose config --quiet
docker compose up -d --wait --wait-timeout 600 model-serving prometheus grafana
docker compose ps
docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
curl -fsS http://127.0.0.1:9090/-/ready
curl -fsS http://127.0.0.1:3000/api/health
python -m src.lk11_check_monitoring --expect targets
```

Untuk menjalankan juga layanan LK09, gunakan `docker compose up -d --build --wait --wait-timeout 600 api db` setelah konfigurasi environment tersedia. Hindari `docker compose down -v` agar volume persisten tidak terhapus.

Di Codespaces, akses port 3000 (Grafana) dan 9090 (Prometheus) melalui forwarded address **Private**. Login Grafana dengan user `admin` dan password lokal `LK11_GRAFANA_PASSWORD` (berlaku saat inisialisasi awal database Grafana). Buka Dashboards → MLOps → LK11 | Jambi Wind Model Observability. Prometheus menampilkan target melalui `/targets` dan status aturan melalui `/alerts`. Jangan membuka port secara publik atau menampilkan password pada screenshot.

## Metrik operasional dan dashboard

| Panel | Sumber dan interpretasi |
|---|---|
| Healthy scrape targets | `sum(up{job="mlflow-model"})`; jumlah target scrape yang berhasil, bukan jaminan setiap request inferensi sukses |
| Drift flags per replica | `lk11_drift_flag`; 0 normal, 1 melampaui ambang, belum tersedia sebelum sampel cukup |
| Throughput per replica | Rate counter `lk11_inference_requests_total` dalam jendela 30 detik, request/s |
| Inference latency p95 | `histogram_quantile(0.95, ...)` dari histogram durasi proxy/backend, jendela 30 detik; estimasi berbasis bucket, bukan latensi jaringan end-to-end klien |
| CPU usage per container | Rate `lk11_container_cpu_seconds_total` dalam 30 detik, dalam satuan core (CPU-second/second), berasal dari cgroup |
| RAM per container | `lk11_container_memory_bytes`, berasal dari cgroup |
| Input wind mean vs synthetic baseline | Rata-rata rolling input dibanding baseline tetap 2,4 m/s |
| Predicted residual (rolling mean) | Rata-rata residual aktual yang dikembalikan model regresi |
| Corrected wind (rolling mean) | Rata-rata `max(0, raw_wind + predicted_residual)` |
| Input shift vs drift threshold | Selisih absolut mean input terhadap baseline dibanding threshold 1 m/s |
| HTTP status by replica | Rate request menurut instance dan status HTTP |
| Rolling window sample count | Jumlah observasi prediksi valid dalam window, maksimal 30 |

Seluruh panel menerima data aktual dalam sesi pengujian. Ketika workload berhenti, throughput dapat menjadi nol dan estimasi p95 dapat menjadi NaN karena tidak ada observasi latensi baru di jendela 30 detik. Rolling window prediksi menyimpan 30 observasi sukses terakhir, bukan window berbasis waktu; nilai dan flag dapat bertahan ketika tidak ada trafik baru. Ini menjelaskan mengapa alert tetap aktif setelah fase shift berakhir hingga sampel recovery masuk.

## Simulasi workload dan drift

Generator `src/lk11_workload.py` mencari port host dinamis menuju port internal 8081 untuk setiap replika. Tiga thread mengirim payload `dataframe_split` satu baris dengan 14 fitur, masing-masing sekitar satu request/s selama 75 detik per fase. Angin sintetis disampling uniform dalam rentang ±0,3 m/s dari mean fase; gust disesuaikan menjadi raw wind + 1,7 m/s. Fitur lain berasal dari contoh LK07. Prediksi diperiksa agar HTTP 200 dan residual finite. Distribusi trafik ditentukan secara eksplisit oleh generator, bukan load balancer otomatis.

```bash
set -o pipefail
python -m src.lk11_workload --phase baseline --seconds 75
python -m src.lk11_check_monitoring --expect baseline
python -m src.lk11_workload --phase shift --seconds 75
python -m src.lk11_check_monitoring --expect shift
python -m src.lk11_workload --phase recovery --seconds 75
python -m src.lk11_check_monitoring --expect recovery
```

Ambil screenshot sebelum beralih ke fase berikutnya, terutama ketika flag dan alert aktif pada fase shift. Verifikasi skrip memeriksa jumlah target UP, keberadaan metrik resource, serta drift flag; keadaan alert dicetak dan diperiksa tersendiri di Prometheus. Untuk reproduksi, pastikan scrape terbaru telah masuk dan durasi evaluasi alert terpenuhi.

Drift detector menggunakan 30 observasi sukses terakhir per replika, minimal 20 observasi sebelum flag diterbitkan. Indikator `D = abs(mean_input - 2.4)` dan flag menjadi 1 jika `D >= 1.0 m/s`. Baseline dan ambang ini khusus demonstrasi sintetis, bukan ambang yang dikalibrasi dari data pelatihan atau observasi cuaca. Fase baseline dan recovery memakai mean 2,4 m/s dan seed yang sama; fase shift memakai mean 5,4 m/s dan seed berbeda.

Alert rules (`monitoring/lk11-rules.yml`):

- `SyntheticInputDrift`: flag 1 bertahan 15 detik.
- `ModelMetricsTargetDown`: target scrape gagal selama 15 detik.
- `ModelReplicaCountBelowThree`: jumlah target UP kurang dari tiga selama 30 detik.

Ketiga aturan lolos `promtool`. Skenario pengujian ini memverifikasi firing dan recovery pada `SyntheticInputDrift`; dua aturan ketersediaan tidak diuji melalui penghentian container. Tidak ada Alertmanager/notifikasi eksternal pada implementasi ini.

## Hasil aktual — 28 November 2026

Startup Compose berhasil; FastAPI dan PostgreSQL tetap healthy, tiga replika MLflow healthy, Prometheus ready dan Grafana melaporkan database `ok`. Prometheus menampilkan tiga target model UP dan satu target internal Prometheus UP. `promtool` menyatakan konfigurasi valid dan menemukan tiga alert rules. Port proxy host saat pengujian: 32772, 32774, 32776. IP scrape: 172.18.0.4:8081, 172.18.0.5:8081, 172.18.0.8:8081. Port dan IP dapat berubah ketika container dibuat ulang.

| Metrik | Baseline | Shift | Recovery |
|---|---:|---:|---:|
| Request sukses (3 replika) | 225 | 225 | 225 |
| Request gagal | 0 | 0 | 0 |
| Counter kumulatif per replika | 75 | 150 | 225 |
| Target model UP | 3/3 | 3/3 | 3/3 |
| Input mean (m/s) | 2,4156–2,4779 | 5,4089–5,4456 | 2,4156–2,4779 |
| Residual mean (m/s) | -0,0808–-0,0782 | -0,5687 | -0,0808–-0,0782 |
| Corrected wind mean (m/s) | 2,3374–2,3971 | 4,8402–4,8769 | 2,3374–2,3971 |
| Input shift (m/s) | 0,0156–0,0779 | 3,0089–3,0456 | 0,0156–0,0779 |
| Window per replika | 30 | 30 | 30 |
| Drift flags | 0/0/0 | 1/1/1 | 0/0/0 |
| Instance alert drift firing | 0 | 3 | 0 |
| Verifikasi | PASS | PASS | PASS |

Total 675 inferensi berhasil dari tiga fase, 225 per replika, tanpa kegagalan pada workload tersebut. Grafik throughput menunjukkan sekitar satu request/s per replika selama trafik stabil. Snapshot akhir recovery menunjukkan rate 0,80–0,92 request/s per replika dan estimasi p95 83,3–91,8 ms. CPU kumulatif 23,34–23,42 CPU-second dan RAM 427.364.352–484.757.504 bytes (sekitar 408–462 MiB) pada snapshot recovery. CPU kumulatif berbeda dari utilisasi CPU pada grafik. Hasil pengujian bersifat fungsional pada beban sintetis ringan; bukan benchmark kapasitas maksimum atau jaminan performa produksi.

## Analisis monitoring dan model decay

Dashboard memperlihatkan bahwa perubahan input sekitar 3 m/s memicu perubahan residual prediksi dan corrected wind serta flag drift pada tiga replika. Alert kembali inactive setelah input normal masuk kembali. Dengan demikian sistem berhasil mendeteksi pergeseran mean input pada skenario yang dirancang dan menyediakan sinyal investigasi proaktif. Nilai residual bukan prediction error atau skor akurasi: perubahan residual sendiri tidak membuktikan model memburuk.

Sesuai rancangan LK01, model decay aktual perlu diperiksa menggunakan reference yang tersedia setelah waktu valid forecast. Pasangkan prediksi dan reference menurut `valid_time` dan horizon, hitung rolling MAE, RMSE, serta bias, lalu bandingkan dengan baseline evaluasi yang setara dan raw ECMWF. Apabila kualitas turun secara konsisten, lakukan investigasi kualitas data, distribusi fitur, dan performa model sebelum memutuskan retraining serta evaluasi challenger. Reference eksperimen sebelumnya adalah ECMWF IFS Analysis, bukan observasi stasiun independen. LK11 belum mengintegrasikan label/reference aktual ke endpoint atau dashboard, belum mengukur model decay aktual, dan belum mengotomatisasi retraining. Detektor mean satu fitur juga tidak mencakup seluruh bentuk data drift atau concept drift.

## Evidence dan luaran

Bukti tekstual disimpan dalam `docs/evidence/lk11/`:

| File | Isi |
|---|---|
| `01-compose-ps.txt` | Status layanan Compose |
| `02-prometheus-targets.txt` | Tiga target dan metrik resource tersedia sebelum workload |
| `03-baseline-workload.txt` | 225 request sukses, 0 gagal |
| `04-baseline-metrics.txt` | Flag 0 pada tiga replika |
| `05-shift-workload.txt` | 225 request sukses, 0 gagal |
| `06-shift-metrics.txt` | Flag 1 dan tiga alert firing |
| `07-recovery-workload.txt` | 225 request sukses, 0 gagal |
| `08-recovery-metrics.txt` | Flag 0, alert kosong, 675 request kumulatif |

Screenshot laporan: Prometheus Target health 3/3 UP, Grafana bagian atas dan bawah pada baseline, shift dan recovery, serta Prometheus Alerts saat firing dan inactive. File konfigurasi utama: `docker-compose.yaml`, `prometheus.yml`, `monitoring/lk11-rules.yml`, dan provisioning dashboard/datasource Grafana. Semua kode dan konfigurasi diperlukan untuk reproduksi; file credential lokal dan model binary tidak disimpan di Git.

Kembali ke [indeks dokumentasi](README.md).
