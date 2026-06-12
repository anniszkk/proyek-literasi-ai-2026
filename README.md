# 📚 Book Diary
### *Leveling Up Your Reading Journey*

Platform rekomendasi bacaan berbasis **Artificial Intelligence** yang membantu pengguna meningkatkan kemampuan literasi secara bertahap melalui analisis kompleksitas buku dan visualisasi Peta Jalan Literasi.

> Proyek Akhir Mata Kuliah Kecerdasan Buatan  
> Program Studi Informatika · Fakultas Sains dan Matematika · Universitas Diponegoro · 2026

---

## 👥 Tim Pengembang — Kelompok Ayam Geprek Kelas C

| Nama | NIM |
|------|-----|
| Annis Fakhiroh Akbar | 24060124130110 |
| Binar Ridha Wiritanaya | 24060124140143 |
| Christianna Olivia J. M. | 24060124140168 |
| Dian Aulya Dewiyani | 24060124130059 |

---

## 🎯 Tentang Proyek

Book Diary menganalisis kompleksitas linguistik buku yang telah dibaca pengguna, lalu merekomendasikan buku berikutnya dengan tingkat kesulitan yang **sedikit lebih tinggi**, sehingga kemampuan literasi meningkat secara bertahap dan terukur.

### Fitur Utama

- 🔬 **Analisis Kompleksitas Bilingual** — skor 0–100 untuk buku bahasa Indonesia dan Inggris menggunakan LFTK + algoritma kustom berbasis morfologi vokal
- 🤖 **Rekomendasi Berbasis AI** — TF-IDF + Cosine Similarity dengan heuristik filter kompleksitas (+3 hingga +15 poin)
- 🗺️ **Peta Jalan Literasi** — visualisasi mind map interaktif (pan, zoom, node-edge kurva Bézier)
- 👤 **Dashboard Profil** — progress bar level literasi, grafik genre favorit, perpustakaan pribadi
- 🔍 **Pencarian Buku** — autocomplete real-time dengan detail sinopsis lengkap

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    BOOK DIARY SYSTEM                    │
├──────────────┬──────────────────────────────────────────┤
│   FRONTEND   │  index.html (HTML + CSS + JS Vanilla)    │
│              │  → Mind Map, Search, Profil, Roadmap     │
├──────────────┼──────────────────────────────────────────┤
│   BACKEND    │  step4_api.py (Flask REST API)           │
│              │  → 5 endpoint JSON                       │
├──────────────┼──────────────────────────────────────────┤
│    MODEL     │  step3_train_recommender.py              │
│              │  → TF-IDF + Cosine Similarity            │
│              │  → Heuristik Peta Jalan (+3~+15 poin)   │
├──────────────┼──────────────────────────────────────────┤
│   ANALISIS   │  step2_complexity.py                     │
│              │  → LFTK (EN) + Vokal Proxy (ID)         │
├──────────────┼──────────────────────────────────────────┤
│     DATA     │  step1_preprocess.py                     │
│              │  → CMU Dataset + Gramedia Scraper        │
└──────────────┴──────────────────────────────────────────┘
```

---

## 📁 Struktur Folder

```
book-diary/
│
├── index.html                    # Frontend (buka langsung di browser)
│
├── step1_preprocess.py           # Download & bersihkan dataset
├── step2_complexity.py           # Hitung skor kompleksitas buku
├── step3_train_recommender.py    # Training model rekomendasi
├── step4_api.py                  # Flask REST API server
│
├── scraper.py                    # Scraper buku Gramedia (Selenium + requests)
├── cleaning.py                   # Membersihkan data
│
├── run_all.py                    # Jalankan step 1–3 sekaligus
├── requirements.txt              # Daftar dependensi Python
├── README.md                     # Dokumentasi ini
│
└── data/                         # Dibuat otomatis saat pipeline dijalankan
    ├── raw/
    │   ├── booksummaries.tar.gz  # Dataset CMU (download otomatis)
    │   └── gramedia_scraped.csv  # Hasil scraping Gramedia
    ├── clean/
    │   ├── books_clean.csv       # Output step1
    │   ├── books_scored.csv      # Output step2 (ada complexity_score)
    │   └── books_features.csv    # Detail fitur linguistik
    └── model/
        └── recommender.pkl       # Model terlatih (output step3)
```

---

## ⚙️ Instalasi & Cara Menjalankan

### Prasyarat
- Python 3.10+
- Google Chrome (untuk scraper Gramedia)
- VSCode atau terminal apapun

### 1. Clone / Download Proyek

```bash
git clone https://github.com/username/book-diary.git
cd book-diary
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Download Resource NLTK (otomatis saat pertama jalan, atau manual)

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 5. Jalankan Pipeline (Step 1–3)

```bash
python run_all.py
```

> Proses ini akan: download dataset CMU (~3MB), menghitung skor kompleksitas semua buku, dan melatih model rekomendasi. Estimasi waktu: **5–15 menit** tergantung spesifikasi laptop.

### 6. Jalankan API Server

```bash
python step4_api.py
```

API akan berjalan di `http://localhost:5000`

### 7. Buka Frontend

Buka file `index.html` langsung di browser (double-click), atau gunakan **Live Server** di VSCode.

---

## 🌐 API Endpoint

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/api/health` | Status server + jumlah buku |
| `GET` | `/api/books/search?q=<query>` | Cari buku berdasarkan judul/penulis |
| `GET` | `/api/books/<book_id>` | Detail lengkap satu buku |
| `POST` | `/api/recommend` | Rekomendasi dari satu buku |
| `POST` | `/api/roadmap` | Peta jalan dari daftar buku |

### Contoh Request `/api/roadmap`

```json
POST http://localhost:5000/api/roadmap
Content-Type: application/json

{
  "read_books": ["Laskar Pelangi", "Perahu Kertas"],
  "top_n": 5
}
```

---

## 🔬 Teknologi yang Digunakan

### Backend & AI
| Teknologi | Fungsi |
|-----------|--------|
| Python 3.10+ | Bahasa pemrograman utama |
| Flask + Flask-CORS | REST API server |
| scikit-learn | TF-IDF Vectorizer + Cosine Similarity |
| LFTK + spaCy | Analisis fitur linguistik (bahasa Inggris) |
| PySastrawi | Stemmer morfologi bahasa Indonesia |
| NLTK | Tokenisasi + stopwords |
| textstat | Readability scoring |
| pandas + numpy | Manipulasi data |
| Selenium | Web scraping |

### Frontend
| Teknologi | Fungsi |
|-----------|--------|
| HTML + CSS + JavaScript | Antarmuka pengguna (vanilla, tanpa framework) |
| Playfair Display + DM Sans | Tipografi (Google Fonts) |
| SVG | Visualisasi mind map (node-edge kurva Bézier) |

### AI Tools yang Membantu Pengembangan
- **Google Gemini** — desain algoritma scraping, preprocessing data, adaptasi LFTK
- **Anthropic Claude** — arsitektur pipeline, REST API, debugging, frontend

---

## 📊 Dataset

| Sumber | Jumlah | Bahasa | Keterangan |
|--------|--------|--------|------------|
| CMU Book Summary Dataset | ~16.000 buku | Inggris | Download otomatis |
| Gramedia Scraper | ~700 buku | Indonesia | Scraping lokal |

---

## 🗺️ Cara Kerja Peta Jalan Literasi

```
Pengguna input buku yang sudah dibaca
            │
            ▼
   Ambil complexity_score tiap buku
            │
            ▼
   Hitung rata-rata → Level saat ini
            │
            ▼
   Cari buku dengan:
   ✓ Genre overlap
   ✓ complexity_score ∈ [score+3, score+15]
            │
            ▼
   Rank by Cosine Similarity (TF-IDF)
            │
            ▼
   Output: Rekomendasi + Visualisasi Mind Map
```

---

## 🐛 Troubleshooting

**API tidak bisa diakses dari browser**
```
Pastikan step4_api.py sudah berjalan dan tidak ada error.
Cek di terminal: "Running on http://0.0.0.0:5000"
```

**Error saat load model (AttributeError pickle)**
```
Pastikan step3_train_recommender.py ada di folder yang sama dengan step4_api.py.
Jangan pindahkan file model ke folder lain.
```

**spaCy model tidak ditemukan**
```bash
python -m spacy download en_core_web_sm
```

**PySastrawi tidak terinstall (stemmer ID fallback ke PorterStemmer)**
```bash
pip install PySastrawi
```

**Scraper Gramedia gagal / kena block**
```
Hentikan scraper, tunggu 1–2 jam, jalankan lagi.
Progress tersimpan otomatis di data/scraped/gramedia_checkpoint.json
```

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis mata kuliah Kecerdasan Buatan,  
Program Studi Informatika, Universitas Diponegoro, 2026.

---

<p align="center">
  Made with ♥ by Kelompok Ayam Geprek Kelas C · Informatika UNDIP 2026
</p>
