# LK04 — Ingestion dan preprocessing yang bisa berjalan otomatis

LK04 menjadi tahap ketika alur data mulai benar-benar dijalankan sebagai pipeline, bukan hanya technical preview. Aku memisahkan proses mengambil forecast dan mengolahnya ke dua skrip, lalu menghubungkannya dengan GitHub Actions.

Sumber data tetap **ECMWF IFS melalui Open-Meteo Single Runs API**, dengan empat target horizon: +6, +12, +18, dan +24 jam.

## Mengambil run yang tersedia

Skrip utama ingestion adalah [`src/ingest_data.py`](../src/ingest_data.py). Skrip ini bisa menerima waktu run tertentu melalui `--run-time`, tetapi untuk eksekusi otomatis ia akan mencari forecast run terbaru yang sudah tersedia.

Pencarian dilakukan dari kandidat run ECMWF pada 00, 06, 12, dan 18 UTC. Mekanisme fallback dibutuhkan karena hasil run terbaru tidak selalu langsung tersedia tepat pada jam inisialisasinya. Request juga dilengkapi timeout dan retry agar gangguan sementara tidak langsung membuat pipeline gagal tanpa percobaan ulang.

Contoh pengujian untuk satu run:

```bash
python src/ingest_data.py --run-time 2026-08-18T00:00
```

File mentah ditulis ke `data/raw/single_runs/` sebagai snapshot per run. File yang sudah ada tidak perlu ditimpa setiap kali pipeline dijalankan ulang.

## Preprocessing dan quality check

Tahap berikutnya dijalankan melalui [`src/preprocess.py`](../src/preprocess.py). Skrip ini membaca raw forecast, memeriksa skema, menyeragamkan tipe timestamp/numerik, menghitung `lead_hours`, lalu memilih empat horizon yang dibutuhkan.

Pemeriksaan penting di tahap ini meliputi:

- kolom yang diperlukan tersedia;
- `run_time` dan `valid_time` dapat dibaca dengan benar;
- data tidak duplikat dan lead-nya sesuai;
- wind speed tidak negatif, arah angin pada 0–360 derajat, serta persentase kelembapan/cloud cover pada rentang yang masuk akal;
- kolom kritis tidak kosong sebelum hasil akhir disimpan.

Arah angin dan waktu diubah menjadi fitur siklik (`sin`/`cos`) agar angka yang berdekatan secara melingkar tidak dianggap terlalu jauh. Timestamp asli tetap disimpan sebagai UTC; konversi ke `Asia/Jakarta` digunakan untuk fitur jam dan hari.

Output utamanya disimpan di `data/processed/`, sedangkan ringkasan pemeriksaan berada di `data/reports/`.

Contoh hasil pengujian historis yang dicatat dalam dokumentasi proyek:

| Pemeriksaan | Hasil |
|---|---:|
| Baris raw forecast | 30 |
| Baris pada empat horizon | 4 |
| Duplikasi | 0 |
| Missing data kritis | 0 |
| Nilai fisik invalid | 0 |
| Baris hasil preprocessing | 4 |
| Kelengkapan data kritis | 100% |

Perlu dicatat bahwa data pada LK04 masih **forecast yang diproses**, belum dataset berlabel final hasil pairing dengan reference seperti yang digunakan pada LK06.

## Menghubungkan ke GitHub Actions

Workflow ada di [`.github/workflows/lk04-data-pipeline.yml`](../.github/workflows/lk04-data-pipeline.yml). Aku menyiapkan tiga jenis pemicu: Pull Request menuju `main`, `workflow_dispatch` untuk menjalankan manual, dan schedule dengan cron `30 */3 * * *` (polling setiap tiga jam).

Di setiap eksekusi, Actions melakukan checkout, menyiapkan Python 3.11, memasang dependency, memeriksa sintaks skrip, menjalankan ingestion, melakukan preprocessing, lalu mengunggah hasil sebagai artifact. Masa simpan artifact pada workflow ini diatur **7 hari**.

```text
GitHub Actions
   → Install dependency
   → Validasi skrip
   → Ingestion latest available forecast
   → Preprocessing
   → Simpan artifact eksekusi
```

Workflow tersebut sudah pernah dijalankan dengan hasil berhasil melalui PR maupun eksekusi manual. Namun, hasil `success` di workflow LK04 **bukan** bukti bahwa model MLflow atau DVC otomatis diuji; pengujiannya memang masih berfokus pada jalur data.

Jejak pengerjaan:

- [PR #5 — Dynamic ingestion dan preprocessing](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/5)
- [PR #6 — Dokumentasi pipeline LK04](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/6)
- [Workflow LK04 di GitHub](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/actions/workflows/lk04-data-pipeline.yml)

## Batasan yang masih ada

Workflow dapat mengambil forecast dan menghasilkan berkas terproses, tetapi belum otomatis menggabungkannya dengan reference terbaru, melatih model baru, atau melakukan deployment. Artifact GitHub Actions juga memiliki masa simpan terbatas dan tidak menggantikan penyimpanan dataset yang berversi.

LK04 membuat tahap pengambilan dan pengolahan forecast lebih konsisten. Persoalan berikutnya adalah **bagaimana melacak perubahan isi dataset**, yang mulai dikerjakan pada LK05.

**Lanjutan:** [LK05 — Dataset versioning](LK05-dataset-versioning.md).
