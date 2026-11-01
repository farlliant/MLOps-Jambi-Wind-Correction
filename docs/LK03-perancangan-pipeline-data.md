# LK03 — Merancang alur data dan mencoba technical preview

Di LK03 aku mulai menerjemahkan rancangan proyek menjadi alur data yang lebih konkret. Yang ingin dipastikan terutama adalah **forecast dan reference bisa dipasangkan pada waktu yang tepat**, data invalid tidak ikut masuk ke training dataset, dan informasi tentang asal forecast tidak hilang setelah preprocessing.

Dokumen LK03 masih berisi banyak keputusan desain. Namun, sudah ada **technical preview** yang dijalankan untuk membuktikan beberapa bagian pentingnya.

## Dua jalur data yang tidak boleh tercampur

Forecast diambil dari **ECMWF IFS melalui Open-Meteo Single Runs API**. Setiap record mempertahankan `run_time`, `valid_time`, dan `lead_hours`, sehingga satu prediksi masih bisa dilacak ke waktu inisialisasi modelnya.

Reference diambil dari **ECMWF IFS Analysis**. Karena data reference baru tersedia setelah waktu yang diprediksi berlalu, jalurnya memang perlu dipisahkan dari ingestion forecast. Forecast baru bisa disimpan dulu, lalu dipasangkan dengan reference ketika data untuk `valid_time` tersebut sudah ada.

Hubungan waktunya:

```text
lead_hours = valid_time - run_time
```

Empat horizon yang dipakai adalah +6, +12, +18, dan +24 jam. Penyimpanan timestamp tetap memakai UTC, sedangkan waktu `Asia/Jakarta` digunakan untuk membuat fitur siklik lokal.

## Alur ETL yang dirancang

```text
Extract forecast ──────┐
                       ├─→ Validasi → Pairing berdasarkan valid_time
Extract reference ─────┘                     ↓
                                      Hitung residual
                                            ↓
                                      Feature engineering
                                            ↓
                                     Processed dataset
                                            ↓
                                      Quality report
```

Pada bagian validasi aku menetapkan beberapa pemeriksaan yang penting untuk data meteorologi: kolom wajib harus tersedia, wind speed tidak boleh negatif, arah angin berada pada 0–360 derajat, timestamp konsisten, dan tidak boleh ada duplikasi berdasarkan natural key.

Untuk forecast, natural key yang direncanakan adalah gabungan `source`, `run_time`, `valid_time`, dan `lead_hours`. Alasan utamanya: **satu `valid_time` bisa diprediksi oleh beberapa forecast run**, sehingga `valid_time` sendiri tidak cukup untuk mengidentifikasi satu prediksi.

Setelah lolos pemeriksaan, data diberi fitur arah angin dalam bentuk `sin` dan `cos`. Fitur waktu lokal juga diubah menjadi `hour_sin`, `hour_cos`, `day_of_year_sin`, dan `day_of_year_cos`. Nilai reference dan residual tidak digunakan sebagai fitur input karena pada saat inferensi nilainya belum tersedia.

## Apa yang sudah diuji?

Technical preview ada di [`src/data/lk03_pipeline_preview.py`](../src/data/lk03_pipeline_preview.py), dengan fungsi akuisisi pendukung dari [`src/data/build_poc_dataset.py`](../src/data/build_poc_dataset.py).

Pengujian dilakukan terhadap forecast run **18 Agustus 2026, 00.00 UTC**. Hasil yang tercatat:

| Pemeriksaan | Hasil |
|---|---:|
| Respons Single Runs API | HTTP 200 |
| Data hourly forecast | 30 baris |
| Horizon +6/+12/+18/+24 | Lengkap, 4 dari 4 |
| Respons reference | HTTP 200 |
| Data reference untuk 18–19 Agustus | 48 baris |
| Pasangan `valid_time` yang ditemukan | 4 dari 4 |
| Paired rows yang lolos | 4 |
| Baris dikeluarkan | 0 |
| Kelengkapan data kritis | 100% |
| Duplikasi key forecast | 0 |

Ada juga pemeriksaan tambahan pada run 18 Agustus 2026 pukul 06.00 UTC. Hasil ini menunjukkan alur dasarnya masuk akal, tetapi masih berupa pengujian skala kecil, belum sistem yang otomatis memproses setiap forecast sepanjang waktu.

Jika data API masih tersedia, preview dapat dicoba dari root repository:

```bash
python -m src.data.lk03_pipeline_preview --run-time 2026-08-18T00:00
```

Perintah tersebut membutuhkan akses API dan akan membuat atau memakai cache data lokal yang tidak semuanya dilacak di Git.

## Penyimpanan dan otomatisasi yang direncanakan

Data dipisah ke lapisan `raw`, `interim`, `processed`, dan `reports`. Saat perancangan, aku juga menyiapkan konsep polling forecast tiap 3 jam, refresh reference harian, retry dengan backoff, serta pengelolaan data invalid melalui quarantine/log.

**Perlu dibedakan antara rancangan dan implementasi:** scheduled workflow GitHub Actions baru diterapkan di LK04. DVC untuk versioning dataset baru diterapkan di LK05. Jadi, dua bagian itu tidak diklaim sebagai fitur yang sudah selesai hanya karena sudah tercantum dalam rancangan LK03.

## Hasil LK03

Hasil paling penting adalah kontrak aliran datanya menjadi lebih jelas: forecast dan reference dipisahkan, pasangan dibuat berdasarkan waktu yang benar, nilai residual dibentuk setelah reference tersedia, dan output validasi dapat diaudit. Technical preview memberi bukti awal sebelum kode ingestion dan preprocessing dibuat lebih mandiri di LK04.

Riwayat implementasi: [PR #4 — Dynamic data pipeline preview](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/4).

**Lanjutan:** [LK04 — Pipeline data dinamis](LK04-pipeline-data-dinamis.md).
