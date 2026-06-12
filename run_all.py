"""
=================================================================
PETA JALAN LITERASI - RUN ALL
=================================================================
Jalankan semua step secara berurutan:
  python run_all.py

Atau jalankan satu per satu:
  python step1_preprocess.py
  python step2_complexity.py
  python step3_train_recommender.py
  python step4_api.py   ← Jalankan ini terpisah (server)
=================================================================

STRUKTUR PROYEK
───────────────
proyek-literasi/
├── requirements.txt          ← Install dulu: pip install -r requirements.txt
├── run_all.py                ← Script ini
│
├── step1_preprocess.py       ← Download + bersihkan data buku
├── step2_complexity.py       ← Hitung Complexity Score (LFTK-inspired)
├── step3_train_recommender.py← Training model rekomendasi
├── step4_api.py              ← Flask API server
│
└── data/
    ├── raw/
    │   └── booksummaries.tar.gz    ← Dataset CMU (download otomatis)
    ├── clean/
    │   ├── books_clean.csv         ← Output step1
    │   ├── books_scored.csv        ← Output step2 (ada complexity_score)
    │   └── books_features.csv      ← Detail fitur linguistik (step2)
    └── model/
        └── recommender.pkl         ← Model terlatih (step3)


ALUR DATA (DATA FLOW)
─────────────────────

[CMU/Dummy Dataset]
       │
       ▼
[Step 1: Preprocess]
  - Download dataset
  - Bersihkan teks
  - Deteksi bahasa ID/EN
  - Output: books_clean.csv
  Kolom: book_id, title, author, genre, language, summary, complexity_score
       │
       ▼
[Step 2: Complexity Score]
  - Ekstrak fitur linguistik (LFTK-inspired):
      flesch_difficulty, gunning_fog, avg_sent_len,
      avg_word_len, ttr, lexical_density, long_word_ratio
  - Hitung weighted score (0–100)
  - Output: books_scored.csv
  Kolom: semua dari step1 + complexity_score + complexity_label
       │
       ▼
[Step 3: Training]
  - TF-IDF pada genre + summary
  - Bangun genre index untuk filter cepat
  - Simpan model ke recommender.pkl
  Input model.recommend(title):
    1. Cari buku sumber
    2. Filter kandidat: genre overlap + complexity [+3, +15]
    3. Rank dengan cosine similarity
    4. Return top-N rekomendasi
       │
       ▼
[Step 4: Flask API]
  - Load model.pkl
  - Expose endpoint REST
  - Format response JSON (lihat kontrak di step4_api.py)
       │
       ▼
[Frontend (future)]
  - Tangkap /api/roadmap
  - Render "Peta Perjalanan" dari roadmap_visualization.nodes + edges
  - Tampilkan progress bar complexity_score
  - Buat koneksi antar buku dari edges


KONTRAK JSON: FIELD PENTING UNTUK FRONTEND
───────────────────────────────────────────
Setiap objek buku dalam response mengandung field berikut.
Penamaan ini HARUS konsisten antara backend dan frontend.

Field              Tipe     Keterangan untuk UI
─────────────────────────────────────────────────────────
book_id            int      React key, link ke halaman detail
title              str      Nama buku, judul kartu
author             str      Nama penulis, sub-judul kartu
genre_list         list     Render sebagai chip/tag berwarna
language           str      Badge "ID" (biru) atau "EN" (hijau)
summary_short      str      Teks 150 karakter, deskripsi kartu
complexity_score   float    Progress bar 0–100
complexity_label   str      Badge warna: hijau/kuning/merah
complexity_delta   float    "+7.3 poin" lebih sulit dari sebelumnya
roadmap_step       str      Teks tooltip di node peta jalan
─────────────────────────────────────────────────────────

FIELD KHUSUS ROADMAP VISUALIZATION (dari /api/roadmap):
  nodes[].type = "read"        → Node buku yang sudah dibaca (biru)
  nodes[].type = "recommended" → Node rekomendasi (oranye)
  edges[].delta                → Angka kenaikan (+) untuk label panah
  complexity_axis.zones        → Warna background zona di sumbu Y


CARA INSTALL & JALANKAN
────────────────────────
1. Buka terminal di VSCode
2. Buat virtual environment (disarankan):
     python -m venv venv
     # Windows:
     venv\Scripts\activate
     # Mac/Linux:
     source venv/bin/activate

3. Install dependencies:
     pip install -r requirements.txt

4. Download model NLTK (otomatis saat pertama jalan, atau manual):
     python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"

5. Jalankan semua pipeline:
     python run_all.py

6. Atau jalankan API saja (jika sudah ada model):
     python step4_api.py

7. Test API di browser atau Postman:
     GET  http://localhost:5000/api/health
     GET  http://localhost:5000/api/books/search?q=harry
     POST http://localhost:5000/api/recommend
          Body: {"book_title": "Harry Potter", "top_n": 5}
     POST http://localhost:5000/api/roadmap
          Body: {"read_books": ["Laskar Pelangi", "Sang Pemimpi"], "top_n": 5}

ESTIMASI RESOURCE (CPU ONLY):
  Step 1: ~10 detik (data dummy) / ~2 menit (dataset penuh)
  Step 2: ~1-3 menit untuk 5000 buku
  Step 3: ~5-15 detik
  Step 4: Response time ~50–200ms per request
  RAM   : ~200-400MB saat API berjalan
=================================================================
"""

import sys
import time

def run_step(step_name: str, func, *args):
    print(f"\n{'='*55}")
    print(f"  MENJALANKAN: {step_name}")
    print(f"{'='*55}")
    t0 = time.time()
    result = func(*args)
    elapsed = time.time() - t0
    print(f"\n[OK] {step_name} selesai dalam {elapsed:.1f} detik")
    return result


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════╗
║      PETA JALAN LITERASI - FULL PIPELINE          ║
║      Book Recommendation AI untuk Pembaca ID/EN   ║
╚═══════════════════════════════════════════════════╝
    """)

    # Import semua step
    try:
        from step1_preprocess       import preprocess_pipeline
        from step2_complexity       import complexity_pipeline
        from step3_train_recommender import training_pipeline
    except ImportError as e:
        print(f"[ERROR] Import gagal: {e}")
        print("        Pastikan semua file step1-3 ada di folder yang sama.")
        sys.exit(1)

    # Jalankan step 1–3
    run_step("Step 1: Pra-pemrosesan Data",    preprocess_pipeline)
    run_step("Step 2: Complexity Scoring",     complexity_pipeline)
    run_step("Step 3: Training Model",         training_pipeline)

    print("""
╔═══════════════════════════════════════════════════╗
║  ✅ SEMUA STEP SELESAI!                           ║
║                                                   ║
║  Sekarang jalankan API server:                    ║
║    python step4_api.py                            ║
║                                                   ║
║  Lalu test di browser:                            ║
║    http://localhost:5000/api/health               ║
╚═══════════════════════════════════════════════════╝
    """)
