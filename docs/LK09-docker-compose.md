# LK09 — Docker Compose dan Multi-Container Orchestration

LK09 melanjutkan implementasi MLOps dari LK06–LK08 dengan menyediakan **API inferensi** dan **database PostgreSQL** sebagai dua layanan yang dikelola melalui Docker Compose. Tujuan utamanya adalah memastikan layanan dapat dijalankan bersama, saling berkomunikasi melalui jaringan internal, serta mempertahankan hasil inferensi meskipun container dibuat ulang.

## Tujuan dan arsitektur

```text
                  Client / curl
                       |
                localhost:8000
                       |
               +----------------+
               | api (FastAPI)  |
               | Model RF-01    |<---- DagsHub MLflow Model Registry
               | inference      |      alias: staging-candidate
               +----------------+      version: 1
                       |
           Docker service discovery: db
                       |
               +----------------+
               | db (PostgreSQL)|
               | inference_     |
               | history       |
               +----------------+
                       |
             Named volume: postgres_data

       Internal bridge network: wind-network
```

- **`api`**: FastAPI menjalankan inferensi residual dan menyediakan endpoint `/health`, `/model`, `/predict`, serta `/history`.
- **`db`**: PostgreSQL 16 Alpine menyimpan riwayat inferensi pada tabel `inference_history`.
- **`wind-network`**: custom bridge network; `api` mengakses database lewat hostname `db`, bukan alamat IP tetap.
- **`postgres_data`**: named volume yang mempertahankan isi database ketika container dihentikan dan dibuat ulang.
- **`depends_on`**: service API baru dijalankan setelah healthcheck database menyatakan `healthy`. API juga memiliki healthcheck sendiri.

Model diperoleh dari **DagsHub MLflow Registry**, bukan disimpan sebagai record di PostgreSQL. API menggunakan model `Jambi-Wind-Residual-RandomForest` dengan alias `staging-candidate` (versi 1 pada pengujian ini). Model pada tahap Staging **tidak otomatis dianggap layak untuk prakiraan operasional**.

## Struktur file LK09

| File | Fungsi |
|---|---|
| `docker-compose.yaml` | Mendefinisikan dua service, jaringan, volume, dependensi, dan healthcheck |
| `Dockerfile` | Membangun image inference API dari Python 3.11 |
| `requirements-lk09.txt` | Dependensi API, PostgreSQL driver, dan kebutuhan model |
| `src/lk09_api.py` | Implementasi FastAPI, loading Registry, inferensi, dan penyimpanan hasil |
| `.dockerignore` | Menghindari penyalinan dataset, environment lokal, serta secret ke image |
| `.env.example` | Template variabel environment tanpa credential asli |
| `.devcontainer/devcontainer.json` | Python 3.11 + Docker-in-Docker pada GitHub Codespaces |

## Menjalankan sistem

Semua perintah berikut dijalankan dari root repository pada environment dengan Docker Engine dan Docker Compose yang aktif.

1. Salin template environment:

   ```bash
   cp .env.example .env
   ```

2. Isi `POSTGRES_PASSWORD` dengan password PostgreSQL buatan sendiri dan `MLFLOW_TRACKING_PASSWORD` dengan access token DagsHub yang memiliki akses ke model. Pastikan URI, username, nama model, dan alias sesuai `.env.example`. **Jangan commit atau mengunggah isi `.env`.**

3. Validasi konfigurasi tanpa mencetak secrets:

   ```bash
   docker compose config --quiet
   ```

4. Build dan jalankan dua service; `--wait` memastikan keduanya healthy sebelum pemeriksaan API:

   ```bash
   docker compose up -d --build --wait --wait-timeout 240
   docker compose ps
   ```

5. Uji endpoint dasar:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   curl -fsS http://127.0.0.1:8000/model
   curl -fsS http://127.0.0.1:8000/history
   ```

Untuk menghentikan sistem **tanpa menghapus data database**:

```bash
docker compose down
```

**Hindari `docker compose down -v`** setelah menyimpan data, karena opsi `-v` menghapus named volume milik proyek Compose beserta isinya.

## Pelaksanaan dan hasil pengujian

Pengujian dilakukan di GitHub Codespaces pada **21 November 2026**.

### 1. Container dan status kesehatan

`docker compose ps` menunjukkan dua container aktif dan berstatus `healthy`:

- `jambi-wind-lk09-api-1` (FastAPI)
- `jambi-wind-lk09-db-1` (PostgreSQL)

Response `/health`:

```json
{
  "status": "healthy",
  "database": "connected",
  "model": "loaded",
  "model_version": "1"
}
```

### 2. Model Registry dan komunikasi jaringan

Endpoint `/model` mengonfirmasi model:

- URI: `models:/Jambi-Wind-Residual-RandomForest@staging-candidate`
- Model version: `1`
- Sumber: DagsHub MLflow Model Registry
- Persetujuan operasional: `not_approved`

Pengujian DNS internal menggunakan `docker compose exec api` membuktikan hostname `db` berhasil di-resolve menjadi `172.18.0.2` pada sesi pengujian. Alamat tersebut bersifat dinamis; yang dipakai oleh aplikasi tetap nama service `db`.

### 3. Inferensi model

Pengujian `POST /predict` menggunakan input **sintetis 14 fitur** yang dihasilkan dari `src/lk07_inference.py` dan mengikuti daftar fitur pada `train.py`. Nilai yang tercatat:

| Parameter | Hasil |
|---|---:|
| Raw forecast | 2.400000000 m/s |
| Predicted residual | -0.083548616 m/s |
| Corrected forecast | 2.316451384 m/s |
| Registered model version | 1 |
| Record ID | 1 |

Rumus koreksi: `max(0, raw forecast + predicted residual)`.

Hasil tersebut hanya membuktikan API dapat melakukan **inferensi teknis**; tidak membuktikan keunggulan akurasi model atas ECMWF pada data observasi independen.

### 4. Penyimpanan dan persistensi PostgreSQL

Setelah inferensi, endpoint `/history` menampilkan `count = 1`, dan query SQL langsung ke tabel `inference_history` menampilkan record `id = 1` dengan raw forecast `2.4` serta hasil koreksi `2.316451383927173`.

Selanjutnya, kedua container dihapus dan dibuat ulang melalui:

```bash
docker compose down
docker compose up -d --wait --wait-timeout 240
```

Setelah dua service kembali `healthy`, endpoint `/history` dan query SQL masih menemukan **record yang sama** (`id = 1`). Ini membuktikan persistensi data menggunakan named volume PostgreSQL pada skenario pengujian tersebut.

Panggilan API segera setelah restart sempat menghasilkan `Connection reset by peer` ketika API masih berstatus `health: starting`. Pengujian ulang **setelah readiness/healthcheck** berhasil; ini menegaskan pentingnya menunggu model selesai dimuat ketika startup.

## Evidence

Bukti pengujian disimpan di direktori [`docs/evidence/lk09/`](evidence/lk09/):

| Nama berkas | Bukti |
|---|---|
| `01-compose-ps.txt` | Status dua container `healthy` |
| `02-service-discovery.txt` | Resolusi DNS `db` dari container API |
| `03-health.json` | Koneksi database dan readiness model |
| `04-model.json` | URI model, alias, dan versi Registry |
| `05-history-after-restart.json` | Record prediksi masih ada setelah container dibuat ulang |
| `06-postgresql-persistence.txt` | Query SQL langsung yang membuktikan record tetap tersimpan |

Untuk laporan akhir, screenshot terminal `docker compose ps` dan pengujian komunikasi antarkontainer dapat dilampirkan bersama evidence tekstual di atas.

## Evaluasi terhadap kebutuhan LK09

- **Struktur Compose**: YAML dengan dua service, build API, healthcheck dan dependensi eksplisit.
- **Integritas sistem**: load model Registry, endpoint sehat, inferensi berhasil, dan komunikasi ke PostgreSQL teruji.
- **Pengelolaan resource**: custom bridge network, named volume, serta data persist setelah recreate container.
- **Reproduksibilitas**: `.env.example` dan instruksi startup melalui satu perintah Compose, dengan syarat token DagsHub valid dan koneksi jaringan tersedia.

## Batasan dan tindak lanjut

- Reference model pada eksperimen berasal dari ECMWF IFS Analysis, bukan observasi stasiun independen.
- Alias `staging-candidate` menunjukkan status akademik/Staging, bukan persetujuan penggunaan operasional.
- Runtime MLflow mencatat perbedaan dependensi opsional (`psutil`, `pytest`, `rich`) dari environment model. Peringatan tersebut tidak menghalangi loading dan inferensi yang telah diuji; penyelarasan dependensi tetap relevan untuk reproduksibilitas lebih ketat.
- Tidak ada autentikasi API pengguna, pengujian beban, monitoring drift, maupun deployment produksi pada LK09 ini.
- Penggunaan token DagsHub perlu mengikuti prinsip least-privilege dan dijaga tetap di luar Git.

Kembali ke [indeks dokumentasi](README.md).
