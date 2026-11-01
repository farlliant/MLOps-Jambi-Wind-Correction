# LK07 — Model Registry, versioning, dan uji inferensi

LK07 melanjutkan eksperimen LK06. Pada tahap ini aku tidak melatih model baru; yang dikerjakan adalah **mengelola model yang sudah ada** di MLflow Model Registry, mencatat asal dataset dan run-nya, mencoba perpindahan stage, lalu memastikan model terdaftar bisa dipanggil untuk inferensi.

Registry yang digunakan bernama `Jambi-Wind-Residual-RandomForest`.

## Dua versi model

Dua konfigurasi LK06 didaftarkan sebagai versi yang berbeda:

| Aspek | Version 1 | Version 2 |
|---|---|---|
| Asal eksperimen | RF-01 | RF-02 |
| MAE validasi | **0,6152 m/s** | 0,6512 m/s |
| RMSE validasi | **0,8084 m/s** | 0,8413 m/s |
| Stage akhir | Production (simulasi) | None |
| Alias | `candidate`, `champion` | `challenger` |
| Keputusan | Dipakai untuk uji inferensi | Pembanding, tidak dipromosikan |

V2 menggunakan artefak RF-02 yang sudah selesai di LK06, bukan training ulang. Run ID, source model, metrik, dan checksum dataset dicocokkan dengan metadata eksperimen sebelumnya.

V1 dipilih sebagai **champion internal** karena lebih baik daripada V2 pada MAE maupun RMSE. Tetapi ada batasan yang tidak boleh diabaikan: **RMSE V1 (0,8084 m/s) masih lebih buruk daripada raw ECMWF baseline (0,7345 m/s)**. Artinya, pemilihan V1 hanya menyatakan versi itu paling baik di antara dua versi yang dibandingkan, bukan sudah memenuhi standar operasional meteorologi.

## Simulasi lifecycle

Aku melakukan simulasi perpindahan stage untuk V1:

```text
None → Staging → Production
```

V2 tetap pada stage `None`. Setelah transisi, alias `champion` diarahkan ke V1, sementara `challenger` tetap ke V2. Alias `candidate` yang diperoleh V1 pada LK06 juga dipertahankan.

MLflow versi 3.17.1 masih menyediakan API stage lama, tetapi API tersebut sudah **deprecated** dan bisa mengeluarkan `FutureWarning`. Untuk memuat model dari program, aku lebih mengandalkan alias modern seperti `@champion`.

Status `Production` pada tugas ini **hanya simulasi akademik**. Registry diberi penanda `deployment_scope=academic_simulation_only` dan `operational_approval=not_approved` agar tidak menimbulkan kesan model telah disetujui untuk sistem nyata.

## Inferensi dari model yang terdaftar

Pengujian dilakukan menggunakan [`src/lk07_inference.py`](../src/lk07_inference.py). Model dimuat melalui URI:

```python
mlflow.pyfunc.load_model(
    "models:/Jambi-Wind-Residual-RandomForest@champion"
)
```

Input pengujian berupa satu contoh forecast **sintetis** dengan 14 fitur yang sama urutannya dengan metadata training. Contoh run menggunakan 29 Oktober 2026 pukul 06.00 UTC, `valid_time` 12.00 UTC, dan horizon +6 jam.

Hasil yang tercatat:

```text
Raw forecast       : 2.4000 m/s
Predicted residual : -0.0835486 m/s
Corrected forecast : 2.3164514 m/s
Model              : V1 / RF-01 (@champion)
```

Perhitungannya adalah `max(0, raw_forecast + predicted_residual)`. Output disimpan di [`data/reports/lk07/inference_champion.json`](../data/reports/lk07/inference_champion.json).

Ini membuktikan jalur **Registry → model loading → feature preparation → prediction** berfungsi pada lingkungan pengujian. Karena inputnya sintetis dan tidak punya nilai reference aktual, hasil tersebut **bukan** bukti ketepatan prakiraan cuaca pada tanggal itu.

## Metadata dan cara memeriksa Registry

Konfigurasi dan lineage dicatat dalam [`config/lk07_model_registry.yaml`](../config/lk07_model_registry.yaml). Isinya antara lain nama Registry, source run, model URI, dua versi model, stage, alias, hasil validasi, pointer DVC, checksum dataset, dan keputusan promosi.

Manajemen Registry ada di [`src/lk07_registry.py`](../src/lk07_registry.py). Jika database dan artefak MLflow dari pengerjaan asli tersedia, kondisi akhir dapat diperiksa tanpa mengubah Registry:

```bash
python src/lk07_registry.py verify
```

Untuk uji inferensi:

```bash
python src/lk07_inference.py --alias champion
```

Skrip Registry juga menyediakan `sync`, `staging`, dan `production`, tetapi operasi yang mengubah Registry membutuhkan `--apply`. **Jangan menjalankan transisi tersebut hanya untuk mengecek hasil**, karena kondisi akhir simulasi sudah dicatat.

Verifikasi `verify` berfokus pada identitas kedua versi, sumber run/model, metrik, stage, alias, checksum training terhadap metadata LK06, dan status simulasi akademik. Audit tambahan di lingkungan pengerjaan juga mencocokkan metadata YAML dengan DVC serta hasil JSON inferensi. Dua jenis pemeriksaan itu dibedakan agar tidak mengklaim semua pengujian dikerjakan oleh satu perintah saja.

## GitHub dan batasan reproduksibilitas

Implementasi LK07 berada pada [PR #10 — Model Registry, versioning, dan inference](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/10), yang telah digabung ke `main`.

Ada hal penting terkait reproduksibilitas: metadata DVC, source code, YAML, dan JSON inferensi sudah ada di Git, tetapi **database MLflow, artefak model, dan cache dataset tidak otomatis ikut ke GitHub**. Pada akhir pengerjaan sudah dibuat backup basis data MLflow dan paket artefak/DVC dari Codespaces. Tanpa file-file tersebut, `git clone` saja belum cukup untuk menjalankan kembali inferensi yang sama dari Registry lokal.

GitHub Actions yang berstatus berhasil pada pengerjaan LK07 merupakan **workflow data pipeline LK04**, bukan pengujian otomatis Registry. Jadi, status CI tersebut tidak dipakai sebagai klaim bahwa seluruh kode LK07 sudah diuji otomatis oleh GitHub Actions.

## Kesimpulan tahap ini

LK07 berhasil melengkapi eksperimen MLflow dengan dua versi model, metadata yang bisa ditelusuri, simulasi lifecycle, dan uji inferensi melalui alias Registry. Bagi pengerjaan MLOps, ini menjadi langkah penting menuju pengelolaan model yang lebih tertib.

Namun, sebelum model benar-benar di-deploy perlu ada penyimpanan artefak persisten, pengujian temporal lebih kuat, data observasi independen, dan kriteria promosi yang mempertimbangkan baseline lebih menyeluruh. **Status Production di LK07 bukan persetujuan deployment nyata.**

Kembali ke [indeks dokumentasi](README.md).
