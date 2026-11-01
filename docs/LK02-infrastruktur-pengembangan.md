# LK02 — Menyiapkan repository dan lingkungan pengembangan

Setelah konsep proyek ditetapkan di LK01, aku lanjut menyiapkan tempat kerja yang bisa dipakai untuk LK berikutnya. Fokus LK02 adalah **membuat struktur repository dan environment yang konsisten**, supaya pengembangan pipeline tidak bergantung pada konfigurasi laptop tertentu.

## Repository dan struktur awal

Repository utama proyek adalah [MLOps-Jambi-Wind-Correction](https://github.com/farlliant/MLOps-Jambi-Wind-Correction). Dari awal aku memisahkan kode, konfigurasi, data, dokumentasi, dan pengujian. Struktur tersebut terus berkembang seiring pengerjaan LK03–LK07.

Bagian-bagian penting yang disiapkan atau dipertahankan dalam repository:

```text
MLOps-Jambi-Wind-Correction/
├── .devcontainer/     # konfigurasi GitHub Codespaces
├── .github/workflows/ # otomasi yang ditambahkan pada LK04
├── config/            # konfigurasi proyek
├── data/              # raw, interim, processed, versioned, reports
├── docs/              # catatan pengerjaan tiap LK
├── models/            # tempat artefak model lokal
├── notebooks/         # eksperimen/notebook jika diperlukan
├── src/               # kode pipeline dan model
├── tests/             # tempat pengujian
├── requirements.txt
└── README.md
```

Tampilan di atas menggambarkan struktur repository **setelah berkembang hingga LK07**. Tidak semua folder atau fitur tersebut sudah memiliki implementasi pada saat LK02 selesai.

## Lingkungan kerja yang dipilih

Pengembangan dilakukan menggunakan **GitHub Codespaces** dengan image Python 3.11 berbasis Debian Bookworm. Konfigurasinya berada di [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json), sedangkan dependency Python dipasang dari [`requirements.txt`](../requirements.txt).

Konfigurasi devcontainer juga menyiapkan extension VS Code seperti Python, Pylance, Jupyter, YAML, dan Ruff. Tujuannya supaya ketika Codespace dibuka, lingkungan dasar yang dibutuhkan sudah tersedia tanpa harus menyiapkan semuanya secara manual.

Untuk memastikan konfigurasi proyek tidak tercecer di dalam kode, parameter lokasi dan sumber data ditaruh di [`config/project.yaml`](../config/project.yaml). Isinya antara lain koordinat Kota Jambi, timezone `Asia/Jakarta`, sumber ECMWF IFS, empat horizon, dan target regresi residual.

## Cara memakai lingkungan ini

Dari root repository, dependency dapat dipasang dengan:

```bash
python -m pip install -r requirements.txt
```

Jika memakai Codespaces, perintah instalasi juga dijalankan otomatis melalui `postCreateCommand` saat container pertama kali dibuat. Untuk pengerjaan lokal, gunakan environment Python yang terpisah agar library proyek tidak bercampur dengan instalasi global.

Alur Git yang digunakan sejak tahap ini adalah membuat branch untuk satu perubahan, melakukan commit, membuka Pull Request, lalu menggabungkannya ke `main` setelah perubahan diperiksa. Jejak LK02 dapat dilihat dari:

- [PR #1 — Standardisasi infrastruktur pengembangan](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/1)
- [PR #2 — Penyesuaian nama/dokumentasi repository](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/2)
- [PR #3 — Pengaturan file artefak yang diabaikan Git](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/3)

## Kenapa beberapa file tidak masuk Git?

Dataset hasil unduhan, keluaran preprocessing, cache, dan artefak model dapat berubah cukup sering atau berukuran besar. Karena itu `.gitignore` digunakan untuk menghindari penyimpanan file operasional tersebut di Git biasa. Kode dan konfigurasi tetap dilacak Git, sedangkan pengelolaan isi dataset dengan DVC baru benar-benar diterapkan pada LK05.

Ini juga berarti **clone repository saja belum tentu cukup untuk menjalankan ulang model LK06–LK07**. Hasil training tetap membutuhkan dataset DVC dan artefak MLflow yang tersedia pada lingkungan asal atau backup-nya. Batasan ini baru terasa penting setelah proyek memasuki tahap training dan Registry.

## Catatan hasil

LK02 membuat fondasi pengembangan yang lebih rapi: repository sudah punya pembagian folder, lingkungan Codespaces bisa digunakan, dan alur branch–PR mulai diterapkan. Tahap ini belum menghasilkan model akhir maupun pipeline data otomatis; fokusnya memang mempersiapkan infrastruktur untuk pengerjaan selanjutnya.

**Lanjutan:** [LK03 — Perancangan pipeline data](LK03-perancangan-pipeline-data.md).
