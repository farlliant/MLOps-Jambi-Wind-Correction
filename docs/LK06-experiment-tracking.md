# LK06 — Training Random Forest dan experiment tracking di MLflow

Di LK06 aku mulai masuk ke bagian machine learning yang lebih lengkap. Fokusnya bukan hanya memperoleh model dengan angka MAE paling kecil, tetapi juga **mencatat eksperimennya dengan rapi**: model dilatih dari data yang mana, memakai parameter apa, hasil evaluasinya berapa, dan artefaknya disimpan di mana.

Aku memakai **MLflow 3.17.1** dan tetap mengerjakan proyek melalui GitHub Codespaces.

## Menyiapkan dataset untuk training

Dataset LK05 masih berisi riwayat prakiraan. Untuk training residual correction, datanya perlu dipasangkan dulu dengan ECMWF IFS Analysis. Aku melakukan historical backfill untuk **1–14 Oktober 2026** pada run 00, 06, 12, dan 18 UTC, dengan tambahan satu run Agustus yang sudah ada sebelumnya.

Hasil pembentukan dataset:

| Tahap | Hasil |
|---|---:|
| Total forecast run | 57 |
| Baris forecast mentah | 1.710 |
| Baris pada horizon +6/+12/+18/+24 | 228 |
| Timestamp reference unik | 63 |
| Sampel berlabel akhir | 228 |
| Baris training | 182 |
| Baris validation | 46 |
| `valid_time` unik pada validation | 13 |

Forecast dipasangkan dengan reference berdasarkan `valid_time`, kemudian target dihitung sebagai:

```text
residual = reference_wind_speed_10m - forecast_wind_speed_10m
```

Model memakai **14 fitur** yang berasal dari forecast, lead time, serta transformasi arah dan waktu. Kolom reference dan residual tidak ikut menjadi fitur prediktor. Untuk menangani fitur numerik yang hilang, pipeline model memakai `SimpleImputer(strategy="median")` yang dipelajari dari data training.

Pembagian data dilakukan **kronologis berdasarkan `valid_time` yang unik**, bukan acak. Sebanyak 50 timestamp pertama masuk training, sedangkan 13 timestamp berikutnya menjadi validation. Ini penting karena beberapa forecast run dapat merujuk ke waktu yang sama; kalau dibagi acak, informasi pada waktu yang sama berpotensi muncul di kedua kelompok.

Data training dan reference dikelola melalui pointer DVC:

- [`data/versioned/lk06_training_dataset.csv.dvc`](../data/versioned/lk06_training_dataset.csv.dvc)
- [`data/versioned/lk06_reference.csv.dvc`](../data/versioned/lk06_reference.csv.dvc)

## Tiga eksperimen yang dijalankan

Training dilakukan melalui [`train.py`](../train.py) dengan `random_state=42`. Aku membandingkan tiga konfigurasi `RandomForestRegressor` pada split validation yang sama.

| Eksperimen | `n_estimators` | `max_depth` | `min_samples_leaf` | MAE validasi | RMSE validasi |
|---|---:|---:|---:|---:|---:|
| RF-01 | 50 | 5 | 2 | **0,6152** | **0,8084** |
| RF-02 | 100 | 10 | 2 | 0,6512 | 0,8413 |
| RF-03 | 200 | 15 | 1 | 0,6960 | 0,8834 |

Semua angka error pada tabel dalam satuan **m/s**. Pembanding awalnya adalah raw forecast ECMWF, dengan **MAE 0,6183 m/s** dan **RMSE 0,7345 m/s**.

RF-01 memiliki MAE paling rendah, bahkan sekitar **0,50% lebih kecil** daripada MAE baseline. Tapi hasil ini perlu dibaca dengan hati-hati: **RMSE RF-01 justru lebih tinggi** daripada baseline. Bias validasi RF-01 juga negatif, sekitar **−0,5551 m/s**. Jadi, aku tidak menyimpulkan bahwa koreksi ini sudah meningkatkan akurasi secara menyeluruh.

Perbandingan per horizon juga disimpan di MLflow. RF-01 memiliki MAE lebih rendah daripada RF-02 dan RF-03 pada keempat horizon, tetapi pengujian terhadap baseline per horizon dan periode waktu yang lebih panjang masih diperlukan.

## Apa yang dicatat di MLflow?

Setiap konfigurasi memiliki run sendiri di experiment `LK06-Jambi-Wind-Residual-Correction`. Run menyimpan hyperparameter, MAE, RMSE, bias, perbaikan MAE dibanding baseline, hasil per horizon, model, dan file prediksi evaluasi.

Skrip juga menghasilkan dua file ringkasan yang dilacak Git:

- [`data/reports/lk06/experiment_comparison.csv`](../data/reports/lk06/experiment_comparison.csv) — ringkasan ketiga run.
- [`data/reports/lk06/best_model_metadata.json`](../data/reports/lk06/best_model_metadata.json) — identitas RF-01, sumber model, fitur, split, metrik, serta checksum SHA-256 dataset.

Model scikit-learn disimpan ke MLflow menggunakan format **skops** beserta signature input. RF-01 kemudian didaftarkan sebagai model **`Jambi-Wind-Residual-RandomForest` versi 1**, dengan alias `candidate` pada tahap LK06. Belum ada klaim bahwa model tersebut layak untuk produksi.

Untuk menjalankan ulang training dari root repository, dataset DVC dan MLflow tracking server lokal harus tersedia lebih dulu:

```bash
python train.py
```

Perintah itu melatih ulang eksperimen dan bisa menghasilkan run serta artefak baru. Jika hanya ingin memeriksa hasil yang sudah tercatat, lebih aman membaca CSV/JSON dan membuka MLflow tanpa menjalankan `train.py` lagi.

## Batasan hasil

Validation hanya mencakup **13 waktu rujukan unik**, dengan satu pembagian temporal dan periode data yang pendek. Reference juga masih berbasis ECMWF IFS Analysis, bukan observasi cuaca independen. Hasilnya cukup untuk membandingkan konfigurasi di LK06, tetapi belum menjadi dasar deployment operasional.

Riwayat pengerjaan ada di [PR #9 — MLflow experiment tracking dan Model Registry](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/9).

**Lanjutan:** [LK07 — Model Registry](LK07-model-registry.md).
