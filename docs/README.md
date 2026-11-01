# Catatan pengembangan per Lembar Kerja

Dokumentasi di folder ini mencatat proses pengerjaan proyek **Jambi Wind Correction MLOps** dari LK01 sampai LK07. Aku pisahkan per tahap supaya README utama tidak terlalu panjang dan keputusan teknis di setiap LK lebih mudah dicari lagi.

| LK | Fokus | Dokumentasi |
|---|---|---|
| 01 | Menentukan masalah, mengecek sumber data, dan menyusun rencana MLOps | [LK01 — Inisiasi proyek](LK01-inisiasi-proyek.md) |
| 02 | Menyiapkan repository dan lingkungan kerja | [LK02 — Infrastruktur pengembangan](LK02-infrastruktur-pengembangan.md) |
| 03 | Merancang pipeline data dan menguji alur awalnya | [LK03 — Perancangan pipeline data](LK03-perancangan-pipeline-data.md) |
| 04 | Implementasi ingestion, preprocessing, dan GitHub Actions | [LK04 — Pipeline data dinamis](LK04-pipeline-data-dinamis.md) |
| 05 | Versioning dataset dengan DVC dan Git | [LK05 — Dataset versioning](LK05-dataset-versioning.md) |
| 06 | Training, eksperimen Random Forest, dan MLflow | [LK06 — Experiment tracking](LK06-experiment-tracking.md) |
| 07 | Model Registry, stage management, dan inferensi | [LK07 — Model Registry](LK07-model-registry.md) |

## Gambaran alurnya

```text
LK01  Studi awal dan perumusan masalah
  ↓
LK02  Repository dan environment
  ↓
LK03  Rancangan + technical preview ETL
  ↓
LK04  Ingestion dan preprocessing otomatis
  ↓
LK05  Riwayat dataset di DVC
  ↓
LK06  Training + MLflow experiment tracking
  ↓
LK07  Model Registry + simulasi lifecycle + inferensi
```

**Yang penting dibedakan:** LK01 dan sebagian LK03 masih berupa studi/rancangan. LK04–LK07 sudah punya implementasi yang dapat ditelusuri ke kode, metadata, dan riwayat Pull Request. Namun, model serving, pemantauan drift secara operasional, dan continuous training otomatis **belum** menjadi fitur yang berjalan end-to-end.

Untuk melihat progres implementasi terbaru, mulai dari [README utama](../README.md), lalu buka dokumentasi LK terkait. Semua contoh perintah dalam folder ini diasumsikan dijalankan dari **root repository**.
