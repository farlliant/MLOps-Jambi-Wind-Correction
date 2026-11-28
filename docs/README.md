# Catatan pengembangan per Lembar Kerja

Dokumentasi di folder ini mencatat proses pengerjaan proyek **Jambi Wind Correction MLOps** dari LK01 sampai LK11. Aku pisahkan per tahap supaya README utama tidak terlalu panjang dan keputusan teknis di setiap LK lebih mudah dicari lagi.

| LK | Fokus | Dokumentasi |
|---|---|---|
| 01 | Menentukan masalah, mengecek sumber data, dan menyusun rencana MLOps | [LK01 — Inisiasi proyek](LK01-inisiasi-proyek.md) |
| 02 | Menyiapkan repository dan lingkungan kerja | [LK02 — Infrastruktur pengembangan](LK02-infrastruktur-pengembangan.md) |
| 03 | Merancang pipeline data dan menguji alur awalnya | [LK03 — Perancangan pipeline data](LK03-perancangan-pipeline-data.md) |
| 04 | Implementasi ingestion, preprocessing, dan GitHub Actions | [LK04 — Pipeline data dinamis](LK04-pipeline-data-dinamis.md) |
| 05 | Versioning dataset dengan DVC dan Git | [LK05 — Dataset versioning](LK05-dataset-versioning.md) |
| 06 | Training, eksperimen Random Forest, dan MLflow | [LK06 — Experiment tracking](LK06-experiment-tracking.md) |
| 07 | Model Registry, stage management, dan inferensi | [LK07 — Model Registry](LK07-model-registry.md) |
| 08 | Otomatisasi testing, training, evaluasi, dan Registry | [LK08 — Code as a Trigger](LK08-code-as-trigger.md) |
| 09 | Docker Compose, API inferensi, PostgreSQL, dan persistensi | [LK09 — Docker Compose](LK09-docker-compose.md) |
| 10 | MLflow REST Serving, Registry demo Production, dan tiga replika | [LK10 — MLflow Model Serving](LK10-mlflow-model-serving.md) |
| 11 | Observabilitas, metrik operasional, dan simulasi drift/recovery | [LK11 — Model Monitoring](LK11-monitoring.md) |

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
  ↓
LK08  Code-triggered testing + retraining + Registry
  ↓
LK09  FastAPI + PostgreSQL melalui Docker Compose
  ↓
LK10  Native MLflow REST Serving + tiga replika
  ↓
LK11  Prometheus + Grafana + simulasi drift/recovery
```

**Yang penting dibedakan:** LK01 dan sebagian LK03 masih berupa studi/rancangan. LK04–LK11 sudah punya implementasi yang dapat ditelusuri ke kode, metadata, dan riwayat Pull Request. API inferensi berjalan melalui Docker Compose pada LK09–LK10; LK11 menambahkan monitoring operasional dan demonstrasi drift input sintetis. Monitoring akurasi berbasis reference aktual dan continuous training otomatis masih menjadi rencana.

Untuk melihat progres implementasi terbaru, mulai dari [README utama](../README.md), lalu buka dokumentasi LK terkait. Semua contoh perintah dalam folder ini diasumsikan dijalankan dari **root repository**.
