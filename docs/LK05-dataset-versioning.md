# LK05 — Mencatat perubahan dataset dengan DVC

Setelah punya pipeline ingestion, aku perlu cara membedakan dataset sebelum dan sesudah penambahan data. Di LK05 aku menggunakan **Data Version Control (DVC)** untuk mengelola isi dataset, sedangkan Git tetap menyimpan kode dan metadata yang menunjuk ke versi data tersebut.

Eksperimennya sengaja sederhana: membuat **Dataset V1**, menambahkan satu siklus forecast menjadi **Dataset V2**, lalu mengecek apakah perubahan itu benar-benar terbaca oleh DVC tanpa membuat duplikasi.

## Pengaturan DVC

DVC diinisialisasi di repository menggunakan `dvc init`. Konfigurasinya terdapat di `.dvc/`, sementara objek dataset disimpan dalam cache DVC lokal. Isi CSV historis tidak dimasukkan langsung ke commit Git; yang dilacak adalah file pointer seperti [`data/versioned/forecast_history.csv.dvc`](../data/versioned/forecast_history.csv.dvc).

Skrip [`src/update_forecast_history.py`](../src/update_forecast_history.py) digunakan untuk menambahkan forecast baru ke dataset historis berdasarkan pasangan waktu yang unik. Tujuannya agar menjalankan skrip dengan input yang sama untuk kedua kalinya tidak menambahkan baris duplikat.

## Dataset V1 dan V2

**V1** berasal dari forecast run 18 Agustus 2026 pukul 00.00 UTC, dengan 30 baris dan 13 kolom. Setelah `dvc add`, metadata awal dicatat pada commit `a64d769`.

**V2** dibuat dengan menambahkan 30 baris forecast run 11 Oktober 2026 pukul 00.00 UTC. Pada percobaan ini run 06.00 UTC belum tersedia, sehingga mekanisme fallback memilih run 00.00 UTC. Dataset kemudian berisi 60 baris tanpa duplikasi pasangan `run_time` dan `valid_time`.

| Aspek | Dataset V1 | Dataset V2 |
|---|---:|---:|
| Forecast run | 1 | 2 |
| Jumlah baris | 30 | 60 |
| Pasangan waktu unik | 30 | 60 |
| Ukuran file | 5.493 byte | 10.815 byte |
| MD5 DVC | `18218cd00751688e36a4688a5bf48072` | `9e01e9a0a05a4302e7aaf18509cd7aa1` |
| Commit metadata | `a64d769` | `f444ba5` |

Ketika data V2 diproses ulang memakai input yang sama, **baris baru yang ditambahkan = 0**. Ini bagian penting dari pengujian idempotency, karena data historis akan terus bertambah dan tidak boleh terduplikasi hanya karena proses dijalankan kembali.

## Cara perubahan diverifikasi

Aku memakai beberapa pemeriksaan yang saling melengkapi:

```bash
dvc status
dvc diff
git diff
git ls-files
git check-ignore
```

`dvc status` digunakan untuk melihat apakah data saat ini masih sesuai dengan metadata. `dvc diff` menunjukkan bahwa dataset berubah antar-revisi, sedangkan pemeriksaan isi CSV diperlukan untuk membuktikan jumlah baris dan duplikasi. Git digunakan untuk melihat perubahan file pointer dan memastikan CSV yang dikelola DVC tidak ikut dilacak sebagai file biasa.

Hasil validasi V2 mencatat **60 baris, 60 pasangan waktu unik, 2 run, dan 0 duplikasi**. Perbedaan checksum MD5 membuktikan isi file V1 dan V2 tidak sama; checksum itu bukan nomor versi yang bertambah secara berurutan.

Riwayat perubahan tersimpan di [PR #8 — DVC dataset versioning](https://github.com/farlliant/MLOps-Jambi-Wind-Correction/pull/8), termasuk commit inisialisasi `4597463`, Dataset V1 `a64d769`, dan Dataset V2 `f444ba5`.

## Hal yang belum selesai

Di LK05 **DVC remote belum dikonfigurasi**. Cache lokal di Codespaces masih menjadi tempat penyimpanan objek dataset. Jadi, orang yang hanya melakukan `git clone` akan memperoleh pointer `.dvc`, tetapi tidak otomatis memperoleh seluruh isi dataset.

Selain itu, dua versi pada LK05 adalah versi **riwayat forecast mentah**, bukan dua versi model dan bukan pula versi akhir dataset training. Dataset berlabel yang menggabungkan forecast dengan reference dibentuk pada LK06 dan juga memperoleh pointer DVC tersendiri.

Buatku hasil LK05 cukup penting karena hubungan antara perubahan kode, metadata data, dan isi dataset sudah mulai bisa diaudit. Tantangan berikutnya bukan lagi sekadar menyimpan data, tetapi melacak eksperimen model yang dilatih dari data itu.

**Lanjutan:** [LK06 — Experiment tracking](LK06-experiment-tracking.md).
