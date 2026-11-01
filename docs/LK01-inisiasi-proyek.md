# LK01 — Menentukan arah proyek Jambi Wind Correction

Di LK01, pekerjaan utamaku bukan langsung membuat model produksi. Aku mulai dari pertanyaan yang lebih mendasar: **apakah prediksi angin dari ECMWF IFS punya kesalahan yang bisa dipelajari, dan bagaimana proses koreksinya nanti dikelola dengan MLOps?**

Aku memilih kasus kecepatan angin 10 meter di **Kota Jambi**, dengan horizon prakiraan +6, +12, +18, dan +24 jam. Ruang lingkup sengaja dibatasi ke satu kota dan satu variabel target supaya eksperimen awalnya tidak terlalu melebar.

## Masalah yang ingin diselesaikan

ECMWF IFS sudah menghasilkan prediksi cuaca berdasarkan pemodelan numerik. Jadi, proyek ini tidak bermaksud membuat model prakiraan cuaca baru dari nol. Ide yang diuji adalah memakai machine learning untuk memprediksi **residual**, yaitu selisih antara data rujukan dan raw forecast.

```text
residual = reference_wind_speed_10m - forecast_wind_speed_10m
corrected_forecast = raw_forecast + predicted_residual
```

Dengan pendekatan ini, informasi dari ECMWF tetap dipakai, lalu model belajar kapan prakiraan awal cenderung terlalu tinggi atau terlalu rendah.

Ada catatan penting tentang sumber label: **ECMWF IFS Analysis bukan pengamatan stasiun cuaca yang independen**. Di tahap awal ini aku memperlakukannya sebagai *reference* berbasis model dan asimilasi data, bukan ground truth observasional.

## Sumber data dan studi awal

Untuk menguji kelayakannya, aku memakai dua sumber dari Open-Meteo:

- **Single Runs API** untuk mengambil prakiraan ECMWF IFS berdasarkan waktu inisialisasi run.
- **Historical Weather API** untuk memperoleh ECMWF IFS Analysis pada waktu yang akan dijadikan rujukan.

Data yang dicek berasal dari **15 Mei sampai 15 Agustus 2026**, dengan siklus forecast 00, 06, 12, dan 18 UTC. Forecast dan reference dipasangkan berdasarkan `valid_time`, bukan hanya berdasarkan urutan baris.

Hasil pemeriksaan awal:

| Pemeriksaan | Hasil |
|---|---:|
| Forecast run yang berhasil diambil | 371 dari 372 |
| Keberhasilan pengambilan | 99,73% |
| Pasangan forecast–reference awal | 1.484 baris |
| Pasangan valid setelah pengecekan | 1.460 baris |
| Kelengkapan data kritis | 98,38% |
| Duplikasi pasangan | 0 |

Baris dengan nilai kritis yang hilang tidak diisi dengan angka perkiraan. Baris tersebut dikeluarkan dari dataset evaluasi agar target residual tidak terbentuk dari data yang sebenarnya tidak tersedia.

Selain mengecek datanya, ada juga eksperimen kelayakan sederhana untuk melihat apakah residual memiliki *learning signal*. Implementasi awal memakai Linear Regression, Random Forest, dan HistGradientBoosting. Hasil yang tercatat pada dokumentasi proyek menunjukkan perbaikan MAE rata-rata sekitar **12,42%** untuk Random Forest pada eksperimen awal. Angka ini adalah hasil studi kelayakan periode Mei–Agustus, **bukan** hasil validasi model LK06 yang memakai dataset dan split berbeda.

Kode terkait yang masih ada di repository:

- [`src/data/build_poc_dataset.py`](../src/data/build_poc_dataset.py) — pengambilan dan penyusunan dataset proof-of-concept.
- [`src/models/sanity_residual_correction.py`](../src/models/sanity_residual_correction.py) — pengujian awal regresi residual.
- [`config/project.yaml`](../config/project.yaml) — konfigurasi lokasi, horizon, sumber forecast, dan target.

## Rancangan MLOps yang disiapkan

Dari LK01 aku menyusun gambaran sistem yang nantinya mencakup:

```text
Ambil forecast → Validasi → Pasangkan dengan reference
               → Bentuk residual → Train & evaluasi model
               → Simpan versi → Serving → Monitoring
               → Retraining jika diperlukan
```

Untuk pembaruan model, rencananya memakai *hybrid trigger*: pemeriksaan terjadwal, penurunan performa, dan perubahan distribusi data. Model baru akan menjadi **challenger** lebih dulu, kemudian dibandingkan dengan **champion** sebelum dipromosikan. Semua ini masih **rancangan** pada LK01, belum otomatis berjalan.

Ukuran keberhasilan yang dipilih adalah MAE sebagai metrik utama, lalu RMSE dan bias sebagai pendukung. Selain itu, kualitas ingestion, kelengkapan data, dan duplikasi juga perlu diperiksa. Tujuannya bukan sekadar memperoleh skor model, tetapi punya pipeline yang bisa ditelusuri saat datanya berubah.

## Hasil LK01 dan batasannya

Dari LK01, aku mendapatkan tiga hal: sumber data bisa diakses, target residual bisa dibentuk dari pasangan waktu yang sesuai, dan rancangan sistem MLOps sudah punya arah yang cukup jelas. Studi awal memberi alasan untuk melanjutkan ke tahap implementasi.

Tetapi hasil awal tersebut belum membuktikan bahwa koreksi ML selalu lebih baik dari raw forecast di semua waktu dan semua horizon. Validasi yang lebih kuat, data observasi independen, dan pengujian pada data terbaru tetap menjadi pekerjaan berikutnya.

**Lanjutan:** [LK02 — Infrastruktur pengembangan](LK02-infrastruktur-pengembangan.md).
