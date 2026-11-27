# LK10 — MLflow Models Serving dan Horizontal Scaling

LK10 melanjutkan orkestrasi LK09 dengan melakukan serving model menggunakan REST API bawaan MLflow Models, membangun image model dengan `mlflow models build-docker`, dan menjalankan tiga replika lewat Docker Compose. Service FastAPI (`api`), PostgreSQL (`db`), bridge network, dan named volume LK09 dipertahankan.

## Tujuan dan batasan

Tujuan LK10 adalah membuktikan model terdaftar dapat melayani permintaan inferensi melalui HTTP dan bahwa tiga container dengan image model yang sama dapat melayani prediksi secara independen. Pengujian memakai data sintetis 14 fitur, bukan evaluasi akurasi model baru. Tiga port replika digunakan secara langsung; belum ada load balancer, uji throughput, atau pembuktian kenaikan kapasitas permintaan.

## Model Registry dan simulasi Production

Model asal di Registry DagsHub adalah `Jambi-Wind-Residual-RandomForest` **Version 2** (run `007309b8db124732804cbdea3acda948`), berstatus **Staging** dan terpilih dengan alias `staging-candidate` saat pengujian LK10. Ini adalah keadaan Registry saat LK10 dan berbeda dari catatan pengujian historis LK09 yang memakai Version 1. LK09 memeriksa stage `Staging` saat startup sehingga model asal **tidak dipromosikan di tempat**.

Untuk memenuhi demonstrasi lifecycle Production, model asal Version 2 disalin ke registered model lain: `Jambi-Wind-Residual-RandomForest-LK10-Demo`, **Version 1, stage Production**. Salinan tersebut dibuat menggunakan `MlflowClient.copy_model_version` lalu `transition_model_version_stage`. Model asal tetap Staging. URI demonstrasi:

`models:/Jambi-Wind-Residual-RandomForest-LK10-Demo/Production`

Stage model Registry sudah dinyatakan *deprecated* pada MLflow modern; contoh Production ini digunakan semata-mata karena persyaratan lembar kerja dan **tidak berarti persetujuan prakiraan cuaca operasional**. Untuk workflow jangka panjang, alias yang eksplisit lebih dianjurkan daripada mengandalkan stage legacy.

## Persiapan lingkungan

Jalankan semua perintah dari root repository GitHub Codespaces. Gunakan Python 3.11, virtual environment proyek, Docker Engine, serta akses DagsHub yang sah. Variabel autentikasi MLflow berasal dari `.env` lokal dan/atau environment; jangan menampilkan maupun commit token. Direktori model lokal `.lk10/` diabaikan Git.

```bash
source .venv/bin/activate
python -m src.lk10_model_ops status
python -m src.lk10_model_ops download
```

Model sumber diunduh ke `.lk10/model`, dan keberadaan `.lk10/model/MLmodel` dapat diperiksa untuk memastikan unduhan sukses.

## MLflow Models REST API lokal

Jalankan model lokal di Terminal 1:

```bash
mlflow models serve \
  --model-uri .lk10/model \
  --port 5001 --host 127.0.0.1 \
  --env-manager local
```

Di Terminal 2, uji endpoint REST API native MLflow (`/ping` dan `/invocations`) melalui skrip yang membentuk payload `dataframe_split`:

```bash
python -m src.lk10_test_serving --url http://127.0.0.1:5001
```

Hasil pengujian aktual: HTTP `200`, raw wind `2.4 m/s`, residual sekitar `-0.083548616 m/s`, dan corrected wind `2.316451383927173 m/s`. Rumus koreksi dilakukan pada sisi klien: `max(0, raw_wind + predicted_residual)`. Native MLflow endpoint mengembalikan residual model, bukan secara otomatis hasil koreksi. Setelah screenshot diambil, server port 5001 dapat dihentikan dengan `Ctrl+C`.

## Docker image dengan MLflow Models

Build image model hasil download:

```bash
mlflow models build-docker \
  --model-uri .lk10/model \
  --name jambi-wind-lk10-mlflow \
  --env-manager virtualenv

docker image ls jambi-wind-lk10-mlflow
```

Image `jambi-wind-lk10-mlflow:latest` berhasil dibangun dan diperiksa dengan `docker image inspect`. Image mengemas artefak model yang diunduh saat build. Docker image tidak disimpan di GitHub; untuk mereproduksinya, pengguna perlu mengunduh model secara sah lalu membangun image ulang. Hasil `docker image ls` di sesi Codespaces menunjukkan disk usage sekitar 5,86 GB, yang bergantung pada cache/image storage Docker setempat.

## Tiga replika pada Docker Compose

Skrip `lk10_apply_compose.py` menambahkan service `model-serving` pada `docker-compose.yaml`, **tanpa menghapus** service `api`, `db`, network `wind-network`, ataupun volume `postgres_data` dari LK09. Konfigurasi menggunakan `deploy.replicas: 3`, image hasil build, healthcheck `/ping`, dan port internal 8080 dengan host port dinamis pada `127.0.0.1`.

```bash
python lk10_apply_compose.py
docker compose config --quiet
docker compose up -d --wait --wait-timeout 600 model-serving
docker compose ps model-serving
python -m src.lk10_test_serving --all-replicas
```

Pada sesi pengujian, tiga container `jambi-wind-lk09-model-serving-{1,2,3}` berstatus `healthy`. Ketiganya dipublikasikan pada host port `32768`, `32769`, dan `32770` saat pengujian. **Port host dapat berubah** setelah container dibuat ulang, jadi jangan mengasumsikan nomor tersebut tetap.

Untuk menemukan port secara dinamis:

```bash
docker compose port --index 1 model-serving 8080
docker compose port --index 2 model-serving 8080
docker compose port --index 3 model-serving 8080
```

Untuk melihat respons masing-masing replika, jalankan `python -m src.lk10_test_serving --all-replicas`. Pada eksekusi yang terdokumentasi, ketiga replika mengembalikan HTTP `200` dan nilai residual serta koreksi yang konsisten untuk input sintetis identik. Perbedaan digit paling akhir residual akibat representasi numerik floating-point tidak bermakna bagi pengujian ini.

### Menyesuaikan jumlah replika secara dinamis

Opsi `--scale` pada Docker Compose bisa digunakan untuk mengubah jumlah replika. Misalnya, lakukan skala turun dari tiga menjadi dua (hemat resource) dan kembali ke tiga:

```bash
docker compose up -d --no-deps --scale model-serving=2 model-serving
docker compose ps model-serving

docker compose up -d --no-deps --scale model-serving=3 model-serving
docker compose up -d --wait --wait-timeout 600 model-serving
docker compose ps model-serving
```

Perintah ini adalah **petunjuk reproduksi**. Bukti utama yang telah diverifikasi adalah tiga replika healthy dan inferensi per-replika; tidak diklaim uji skala 2→3 sudah dijalankan kecuali evidence baru diambil. Perintah `up -d` tanpa `--scale` menggunakan konfigurasi tiga replika dari YAML.

## Serving Registry Production sebagai simulasi akademik

Operasi registrasi dilakukan menggunakan file `src/lk10_registry_demo.py`. Gunakan perintah status untuk membaca metadata model, `prepare` hanya jika salinan demo belum ada, dan `serve` untuk menguji serving model salinan Production:

```bash
python -m src.lk10_registry_demo status
# Hanya bila salinan demo belum dibuat:
# python -m src.lk10_registry_demo prepare

# Terminal 1
python -m src.lk10_registry_demo serve

# Terminal 2
python -m src.lk10_test_serving --url http://127.0.0.1:5002
```

Pada pengujian, URI Production demo berhasil dimuat melalui `mlflow models serve` pada port 5002. Endpoint `/ping` dan `/invocations` merespons HTTP `200`, dengan hasil corrected wind `2.316451383927173 m/s` untuk input yang sama.

**Pembedaan penting:** tiga replika Docker menggunakan image yang dibangun dari artefak model asal **Version 2 Staging**, sedangkan port 5002 melayani model dari registered model **Production demonstrasi** yang merupakan salinan sumber Version 2. Jadi Production pada Registry telah diuji secara terpisah, bukan mengklaim image tiga replika dibangun langsung dari URI Production.

## Evidence yang telah diverifikasi

Lima file disimpan pada [`docs/evidence/lk10/`](evidence/lk10/):

| File | Isi |
|---|---|
| `01-three-replicas.txt` | Tiga container MLflow berstatus healthy |
| `02-replica-inference.txt` | HTTP 200 dari ketiga replika dan hasil konsisten |
| `03-registry-status.txt` | Model sumber Version 2 pada Staging |
| `04-production-demo.txt` | Salinan Registry demo Version 1 pada Production |
| `05-production-rest-inference.txt` | HTTP 200 dan prediksi dari MLflow Production demo pada port 5002 |

Screenshot terminal yang relevan: hasil REST API lokal, `docker compose ps model-serving`, hasil inferensi tiga replika, dan serving Production demo. Model, dataset, maupun nilai uji tidak bersumber dari pengamatan real-time cuaca.

## Evaluasi terhadap rubrik LK10

- **Kualitas inference API:** serving native MLflow menerima permintaan JSON dan mengembalikan prediksi residual dengan HTTP 200.
- **Implementasi skalabilitas:** tiga replika sehat dengan port host berbeda; setiap replika berhasil memproses input yang sama.
- **Efisiensi alur kerja:** image dibangun menggunakan perintah MLflow dan jumlah replika diatur dalam Docker Compose.
- **Dokumentasi replikasi:** README dan dokumen ini menyediakan cara menjalankan model, menemukan endpoint replika, menguji prediksi, dan mengubah skala layanan.

## Batasan

Status Production pada model `-LK10-Demo` hanya simulasi untuk memenuhi rubrik mata kuliah; model asal tetap Staging agar LK09 tidak terganggu. Ketiga replika menggunakan model yang sama dan belum diuji melalui load balancer atau benchmark request konkuren. Evaluasi model awal menggunakan ECMWF IFS Analysis sebagai rujukan, bukan observasi stasiun independen. Akses API dibatasi ke loopback Codespaces; belum ada autentikasi publik, monitoring drift, atau mekanisme autoscaling berbasis metrik.

Kembali ke [indeks dokumentasi](README.md).
