# LK08 — Code as a Trigger dan Otomatisasi MLOps

LK08 melanjutkan implementasi LK06 dan LK07 dengan
menghubungkan automated testing, retraining, evaluasi
model, dan registrasi ke dalam GitHub Actions.

Tujuan utamanya adalah memastikan perubahan kode
memicu pemeriksaan otomatis sebelum model baru
boleh masuk MLflow Model Registry.

## Mekanisme trigger

Workflow berada di
`.github/workflows/mlops-automation.yaml`.

Dua event utama yang digunakan:

- `pull_request` menuju `main`: menjalankan automated
  testing tanpa akses ke credential Registry.
- `push` ke `main`: menjalankan testing, DVC pull,
  retraining, evaluasi, dan registrasi Staging jika lolos.

Pemisahan tersebut dilakukan agar kode Pull Request
tidak langsung memperoleh akses tulis ke Model Registry.

## Automated testing

Pengujian menggunakan pytest dan mencakup:

- Validation gate MAE, RMSE, dan absolute bias.
- Penolakan metrik NaN dan Infinity.
- Kelengkapan empat forecast horizon.
- Validasi residual reference dikurangi forecast.
- Penolakan duplicate forecast key.
- Penolakan missing value pada kolom penting.
- Temporal split tanpa overlapping valid_time.

Pada pemeriksaan lokal awal, seluruh 16 tes berhasil.

## Reproducible dataset

Dataset training dipulihkan dari remote DVC DagsHub
menggunakan pointer
`data/versioned/lk06_training_dataset.csv.dvc`.

Dataset referensi LK06 memiliki 228 baris,
dengan 182 baris training dan 46 baris validation.

SHA-256 dataset awal:
`59ee06bb1b873097237a15ad755c3ce561df7feda28095cf933922124d0a2436`

Dataset historis hanya digunakan sebagai pembanding
apabila identitas dataset dan temporal split konsisten.

## Automated training

Training tetap menggunakan `train.py` dari LK06,
dengan tiga konfigurasi Random Forest:

- RF-01
- RF-02
- RF-03

Perubahan LK08 pada `train.py` hanya menambahkan
konfigurasi melalui environment variable untuk
tracking URI, nama eksperimen, dan lokasi laporan.

Model dan metrik disimpan melalui MLflow remote DagsHub.

## Automated evaluation

Kandidat terbaik dibandingkan dengan raw ECMWF
dan model pembanding.

Validation gate memeriksa:

- MAE tidak lebih buruk daripada raw ECMWF.
- MAE tidak lebih buruk daripada model pembanding.
- RMSE tidak lebih buruk daripada model pembanding.
- Absolute bias tidak lebih buruk daripada pembanding.
- Kelengkapan horizon dan waktu validasi.
- Integritas dataset, metrik, dan MLflow run.

Jika champion tersedia, perbandingan dilakukan
pada validation holdout yang sama.

Jika belum tersedia, benchmark historis RF-01
hanya diperbolehkan untuk dataset LK06 yang identik.

## Auto-registration

Model yang lolos evaluasi didaftarkan ke
`Jambi-Wind-Residual-RandomForest` dengan stage Staging.

Registrasi tidak otomatis mengganti alias `champion`
dan tidak berarti model sudah disetujui untuk
operasional prakiraan cuaca.

Stage Staging digunakan untuk mengikuti instruksi
akademik LK08, meskipun API stages MLflow sudah
deprecated.

## Pencegahan silent failures

Otomatisasi tidak cukup hanya memastikan kode
berhasil dieksekusi. Kualitas data, konsistensi
residual, integritas waktu, dan performa model
juga harus diperiksa.

Jika salah satu pemeriksaan wajib gagal, proses
berhenti sebelum model diregistrasikan.

## Bukti pelaksanaan

Bagian ini akan dilengkapi setelah GitHub Actions
berhasil dijalankan:

- URL Pull Request:
- URL workflow run:
- Commit SHA:
- Hasil automated testing:
- Hasil DVC pull:
- MLflow experiment dan run ID:
- Hasil validation gate:
- Registered model version:
- Status Staging:

## Batasan

Reference yang digunakan adalah ECMWF IFS Analysis,
bukan observasi stasiun independen.

Otomatisasi performance degradation, data drift,
dan scheduled model retraining masih menjadi
rencana pengembangan hybrid trigger dari LK01.

Status Staging bersifat simulasi akademik,
bukan persetujuan deployment operasional.

Kembali ke [indeks dokumentasi](README.md).
