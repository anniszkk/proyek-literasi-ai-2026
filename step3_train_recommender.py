"""
=================================================================
PETA JALAN LITERASI - STEP 3: TRAINING MODEL REKOMENDASI
=================================================================
Diadaptasi dari:
  Reinalynn/Building-a-Book-Recommendation-System-using-Python
  └─ Content filtering by vectorizing on full text (tfidf and count).ipynb

Perbedaan utama dari repo asli Reinalynn:
  ┌─────────────────────────────────────────────────────────────┐
  │ REPO ASLI (Reinalynn)         │ ADAPTASI KITA               │
  ├───────────────────────────────┼─────────────────────────────┤
  │ Vectorize "full_text" =       │ Vectorize "content_text" =  │
  │ tahun + judul + review user   │ genre (3x) + summary buku   │
  ├───────────────────────────────┼─────────────────────────────┤
  │ TF-IDF → cosine similarity    │ SAMA persis (kode diadopsi) │
  │ → cari buku PALING mirip      │ TAPI: filter complexity     │
  ├───────────────────────────────┼─────────────────────────────┤
  │ Target: similarity = 100%     │ Target: genre mirip,        │
  │ (maximum match)               │ complexity +3 s/d +15 poin  │
  ├───────────────────────────────┼─────────────────────────────┤
  │ Dataset: Goodreads 10k buku   │ Dataset: CMU/dummy ID+EN    │
  │ Input: review_text pengguna   │ Input: summary sinopsis buku│
  ├───────────────────────────────┼─────────────────────────────┤
  │ Bahasa: English saja          │ Bahasa: Indonesia + English │
  │ Lemmatizer: WordNetLemmatizer │ Lemmatizer: PorterStemmer   │
  │ (English-specific)            │ (lebih general, cocok ID)   │
  └───────────────────────────────┴─────────────────────────────┘

Pipeline Reinalynn yang diadopsi (dari notebook-nya):
  1. Load CSV → kolom "full_text" (kita: "content_text")
  2. TfidfVectorizer().fit_transform(ds['full_text'])
  3. cosine_similarity(tfidf_matrix, tfidf_matrix)
  4. Ambil index buku dengan similarity tertinggi
  → Kita modifikasi langkah 4: filter complexity dulu, baru rank

Jalankan: python step3_train_recommender.py
=================================================================
"""

import os
import re
import pickle
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer          # Reinalynn pakai WordNetLemmatizer (EN only)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Tambahan baru
                                              # Kita ganti PorterStemmer agar support ID juga
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inisiasi dua mesin stemmer yang berbeda
stemmer_en = PorterStemmer()
stemmer_id = StemmerFactory().create_stemmer()

# ─── PATH ───────────────────────────────────────────────────────
SCORED_FILE = os.path.join("data", "clean", "books_scored.csv")
MODEL_DIR   = os.path.join("data", "model")
MODEL_FILE  = os.path.join(MODEL_DIR, "recommender.pkl")

# ─── HEURISTIK "PETA JALAN" ─────────────────────────────────────
# Ini adalah modifikasi utama dari logika Reinalynn.
# Reinalynn mencari yang PALING mirip (similarity tertinggi).
# Kita mencari yang genre mirip TAPI complexity sedikit lebih tinggi.
COMPLEXITY_MIN_DELTA = 3    # minimal +3 poin dari buku sebelumnya
COMPLEXITY_MAX_DELTA = 15   # maksimal +15 poin (jangan terlalu jauh)
TOP_N_DEFAULT        = 5


# ─── TEXT PREPROCESSING ─────────────────────────────────────────
# Diadaptasi dari fungsi clean_txt() di notebook Reinalynn.
# Reinalynn mendefinisikan black_txt() + clean_txt() tapi akhirnya
# memilih TIDAK menggunakannya (cell berisi "# decided not to use").
# Kita tetap menggunakannya karena dataset kita lebih bervariasi
# (campuran ID+EN), dan mengganti lemmatizer ke stemmer.

def setup_nltk():
    for path, name in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# Stopword bahasa Indonesia (manual, NLTK tidak punya korpus ID)
STOPWORDS_ID = {
    "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "dari",
    "ke", "ada", "tidak", "juga", "sudah", "akan", "ia", "mereka",
    "kita", "pada", "oleh", "dalam", "adalah", "tersebut", "saat",
    "jika", "maka", "namun", "tetapi", "karena", "seperti", "bahwa",
    "atau", "pun", "lalu", "kemudian", "ketika", "setelah", "sebelum",
    "antara", "hingga", "sampai", "meski", "walaupun", "agar",
}


def get_stopwords(lang: str) -> set:
    """
    Kembalikan set stopword sesuai bahasa.
    Dipisah per bahasa agar kata seperti "di" tidak ikut
    dibuang saat membersihkan teks bahasa Inggris.

    lang="en" → stopwords NLTK English
    lang="id" → stopword Indonesia manual (NLTK tidak punya korpus ID)
    """
    if lang == "en":
        return set(stopwords.words("english"))
    return STOPWORDS_ID


stemmer = PorterStemmer()


def black_txt(token: str, stop_words: set) -> bool:
    """
    Filter token: bukan stopword, bukan tanda baca, panjang > 2.
    Diadaptasi langsung dari fungsi black_txt() Reinalynn.
    """
    return (
        token not in stop_words and
        token not in list(string.punctuation) and
        len(token) > 2
    )


def clean_txt(text: str, lang: str) -> str:
    """
    Bersihkan teks dengan Stemmer Hibrida (PySastrawi untuk ID, Porter untuk EN).
    Stopword akan ditarik secara dinamis berdasarkan parameter 'lang'.
    """
    if not isinstance(text, str):
        return ""

    # Tarik stopword sesuai bahasa secara dinamis
    stop_words = get_stopwords(lang)

    text = re.sub("'", "", text)
    text = re.sub(r"(\d|\W)+", " ", text)
    text = text.replace("nbsp", "")

    tokens = text.lower().split()
    clean  = []
    
    for w in tokens:
        if black_txt(w, stop_words):
            # Cek bahasa untuk menentukan alat potong (stemmer)
            if lang == "id":
                clean.append(stemmer_id.stem(w))
            else:
                clean.append(stemmer_en.stem(w))

    return " ".join(clean)


def build_full_text(row: pd.Series) -> str:
    """
    Bangun "full_text" untuk setiap buku — analog dengan kolom
    "full_text" di dataset Reinalynn yang berisi gabungan metadata.

    Reinalynn: full_text = tahun + judul + review_text pengguna
    Kita      : full_text = genre (bobot 3x) + summary sinopsis

    Genre diulang 3x agar lebih berbobot dalam TF-IDF,
    mengikuti prinsip field weighting yang Reinalynn terapkan
    dengan menyertakan judul (yang juga berulang via metadata).
    """
    genre   = str(row.get("genre",   "")).replace("|", " ")
    summary = str(row.get("summary", ""))
    return f"{genre} {genre} {genre} {summary}"



# ─── MODEL KELAS ────────────────────────────────────────────────

class LiteracyRoadmapRecommender:
    """
    Model Peta Jalan Literasi berbasis Content-Based Filtering.

    Fondasi: TF-IDF + cosine_similarity dari Reinalynn
    Modifikasi: filter complexity "sedikit lebih sulit" untuk
                menghasilkan peta jalan literasi bertahap.
    """

    def __init__(self):
        self.df_books         = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix     = None
        self.cosine_sim       = None   # Reinalynn hitung full matrix — kita simpan juga
        self.genre_index      = {}

    # ── TRAINING ──────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        """
        Latih model dari DataFrame buku.
        Mengikuti alur notebook Reinalynn:
          ds['full_text'] → TfidfVectorizer → cosine_similarity matrix
        """
        print("[MODEL] Memulai training...")
        setup_nltk()

        self.df_books = df.copy().reset_index(drop=True)
        self.df_books["complexity_score"] = (
            self.df_books["complexity_score"].fillna(50.0).astype(float)
        )

        # ── Step 1: Bangun full_text (analog kolom Reinalynn) ────
        print("[MODEL] Membangun full_text (genre × 3 + summary)...")
        self.df_books["full_text"] = self.df_books.apply(build_full_text, axis=1)

        # ── Step 2: Bersihkan teks per bahasa ───────────────────
        # Reinalynn mendefinisikan clean_txt() tapi memilih skip.
        # Kita tetap pakai karena dataset campuran ID+EN perlu normalisasi.
        # Stopword dipilih per buku sesuai kolom "language" agar
        # kata seperti "di" tidak ikut dibuang dari teks bahasa Inggris.
        # Tambahkan parameter axis=1 dan panggil row["language"]
        self.df_books["full_text_clean"] = self.df_books.apply(
            lambda row: clean_txt(row["full_text"], row["language"]),
            axis=1
        )

        # ── Step 3: TF-IDF Vectorizer ────────────────────────────
        # Persis dari notebook Reinalynn:
        #   TfidfVectorizer()
        #   tfidf_matrix = tfidf.fit_transform(ds['full_text'])
        # Kita tambah max_features dan ngram_range agar ringan di CPU
        print("[MODEL] Fitting TF-IDF Vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,     # Reinalynn tidak batasi (dataset-nya 10k buku)
            ngram_range=(1, 2),    # unigram + bigram, lebih kaya dari pure unigram
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df_books["full_text_clean"]
        )
        print(f"         TF-IDF matrix: {self.tfidf_matrix.shape}")

        # ── Step 4: Cosine Similarity Matrix ─────────────────────
        # Persis dari Reinalynn:
        #   cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # Catatan: untuk dataset besar ini bisa berat di RAM.
        # Kita simpan full matrix hanya jika < 3000 buku,
        # sisanya hitung on-demand saat query.
        n_books = len(self.df_books)
        if n_books <= 3000:
            print("[MODEL] Menghitung cosine similarity matrix (full)...")
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            print(f"         Cosine sim matrix: {self.cosine_sim.shape}")
        else:
            print(f"[MODEL] Dataset besar ({n_books} buku) → cosine similarity on-demand")
            self.cosine_sim = None

        # ── Step 5: Genre Index ───────────────────────────────────
        print("[MODEL] Membangun genre index...")
        for idx, row in self.df_books.iterrows():
            for g in str(row["genre"]).split("|"):
                g = g.strip().lower()
                if g:
                    self.genre_index.setdefault(g, []).append(idx)

        print(f"         Genre unik: {len(self.genre_index)}")
        print("[MODEL] Training selesai ✓")
        return self

    # ── INFERENSI ─────────────────────────────────────────────────

    def _get_book_idx(self, title: str) -> int:
        """Cari index buku berdasarkan judul (partial match, case-insensitive)."""
        mask = self.df_books["title"].str.lower().str.contains(
            title.lower().strip(), na=False
        )
        matches = self.df_books[mask]
        if matches.empty:
            raise ValueError(f"Buku '{title}' tidak ditemukan dalam dataset.")
        return matches.index[0]

    def _get_similarity_scores(self, source_idx: int, candidate_indices: list) -> np.ndarray:
        """
        Hitung cosine similarity antara buku sumber dan kandidat.
        Jika full matrix sudah dihitung (Reinalynn style), pakai itu.
        Jika tidak, hitung on-demand untuk kandidat saja (hemat RAM).
        """
        if self.cosine_sim is not None:
            # Reinalynn: cosine_sim[idx] → similarity semua buku dengan buku idx
            return self.cosine_sim[source_idx][candidate_indices]
        else:
            # On-demand: hitung hanya untuk kandidat
            source_vec    = self.tfidf_matrix[source_idx]
            candidate_mat = self.tfidf_matrix[candidate_indices]
            return cosine_similarity(source_vec, candidate_mat)[0]

    def _get_candidates(self, source_idx: int, complexity_score: float) -> list:
        """
        Modifikasi utama dari logika Reinalynn:
        Reinalynn: ambil semua buku, rank by similarity, return top-N
        Kita      : filter genre overlap + complexity zone, BARU rank by similarity

        Zona complexity: [score + MIN_DELTA, score + MAX_DELTA]
        """
        source_genres = {
            g.strip().lower()
            for g in str(self.df_books.loc[source_idx, "genre"]).split("|")
        }

        # Kumpulkan kandidat bergenre overlap
        genre_candidates = set()
        for g in source_genres:
            genre_candidates.update(self.genre_index.get(g, []))
        genre_candidates.discard(source_idx)

        # Filter complexity zone
        lo = complexity_score + COMPLEXITY_MIN_DELTA
        hi = complexity_score + COMPLEXITY_MAX_DELTA
        primary = [
            i for i in genre_candidates
            if lo <= self.df_books.loc[i, "complexity_score"] <= hi
        ]

        # Fallback: perlebar zona jika kurang dari 3 kandidat
        if len(primary) < 3:
            lo_fb, hi_fb = complexity_score + 1, complexity_score + 25
            fallback = [
                i for i in genre_candidates
                if lo_fb <= self.df_books.loc[i, "complexity_score"] <= hi_fb
                and i not in primary
            ]
            primary += fallback

        return primary

    def recommend(self, book_title: str, top_n: int = TOP_N_DEFAULT) -> list:
        """
        Rekomendasikan buku berikutnya di peta jalan literasi.

        Alur (modifikasi dari Reinalynn):
          1. Cari index buku sumber (sama dengan Reinalynn)
          2. Filter kandidat: genre overlap + complexity zone   ← BARU
          3. Hitung cosine_similarity untuk kandidat saja       ← dioptimasi
          4. Return top_n terurut by similarity                 ← sama

        Reinalynn langsung ke step 3 tanpa filter,
        sehingga buku yang paling mirip bisa saja level yang sama.
        """
        if self.df_books is None:
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        source_idx   = self._get_book_idx(book_title)
        source_row   = self.df_books.loc[source_idx]
        source_score = float(source_row["complexity_score"])

        candidates = self._get_candidates(source_idx, source_score)
        if not candidates:
            return []

        sims   = self._get_similarity_scores(source_idx, candidates)
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)

        return [
            self._format_book(self.df_books.loc[idx], sim, source_score)
            for idx, sim in ranked[:top_n]
        ]

    def get_roadmap(self, read_books: list, top_n: int = TOP_N_DEFAULT) -> dict:
        """Buat peta jalan dari daftar buku yang sudah dibaca."""
        source_data, all_recs, scores = [], [], []

        for title in read_books:
            try:
                idx   = self._get_book_idx(title)
                row   = self.df_books.loc[idx]
                recs  = self.recommend(title, top_n=top_n)

                source_data.append(self._format_book(row))
                all_recs.extend(recs)
                scores.append(float(row["complexity_score"]))
            except ValueError as e:
                print(f"[WARN] {e}")

        if not scores:
            return {"error": "Tidak ada buku yang ditemukan."}

        current_level = round(float(np.mean(scores)), 1)

        # Deduplikasi, prioritas similarity tertinggi
        seen, unique_recs = set(), []
        for rec in sorted(all_recs, key=lambda x: x["similarity_score"], reverse=True):
            if rec["book_id"] not in seen:
                seen.add(rec["book_id"])
                unique_recs.append(rec)

        return {
            "current_level":   current_level,
            "current_label":   get_complexity_label(current_level),
            "target_range": {
                "min": round(current_level + COMPLEXITY_MIN_DELTA, 1),
                "max": round(current_level + COMPLEXITY_MAX_DELTA, 1),
            },
            "source_books":    source_data,
            "recommendations": unique_recs[:top_n * 2],
        }

    def _format_book(self, row, similarity_score=None, source_score=None) -> dict:
        """
        Format satu buku menjadi dict JSON-ready.
        Nama field ini adalah KONTRAK dengan frontend — jangan diubah
        tanpa update dokumentasi di step4_api.py.
        """
        c_score = float(row.get("complexity_score", 50))
        genres  = [g.strip() for g in str(row.get("genre", "")).split("|") if g.strip()]
        summary = str(row.get("summary", ""))
        delta   = round(c_score - source_score, 1) if source_score is not None else None

        return {
            "book_id":          int(row.get("book_id", 0)),
            "title":            str(row.get("title", "")),
            "author":           str(row.get("author", "")),
            "genre":            str(row.get("genre", "")),
            "genre_list":       genres,
            "language":         str(row.get("language", "en")),
            "summary":          summary,
            "summary_short":    summary[:150] + "..." if len(summary) > 150 else summary,
            "complexity_score": round(c_score, 1),
            "complexity_label": get_complexity_label(c_score),
            "complexity_delta": delta,
            "similarity_score": round(float(similarity_score), 3) if similarity_score is not None else None,
            "roadmap_step":     f"Langkah berikutnya (+{delta:.0f} poin)" if delta is not None else None,
        }


def get_complexity_label(score: float) -> str:
    if score <= 20:   return "Sangat Mudah"
    elif score <= 40: return "Mudah"
    elif score <= 60: return "Menengah"
    elif score <= 80: return "Sulit"
    else:             return "Sangat Sulit"


# ─── PIPELINE ───────────────────────────────────────────────────

def training_pipeline():
    print("\n" + "="*60)
    print("  PETA JALAN LITERASI - STEP 3: TRAINING MODEL")
    print("  Diadaptasi dari: Reinalynn/Building-a-Book-Recommendation")
    print("  Modifikasi: genre filter + complexity heuristic")
    print("="*60)

    if not os.path.exists(SCORED_FILE):
        print(f"[ERROR] {SCORED_FILE} tidak ditemukan.")
        print("        Jalankan step2_complexity.py terlebih dahulu!")
        return None

    df = pd.read_csv(SCORED_FILE, encoding="utf-8-sig")
    print(f"[OK] Loaded {len(df)} buku")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = LiteracyRoadmapRecommender()
    model.fit(df)

    # Simpan model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[SUKSES] Model disimpan: {MODEL_FILE}")

    # ── TEST ─────────────────────────────────────────────────────
    print("\n--- TEST REKOMENDASI ---")
    for title in ["Laskar Pelangi", "Harry Potter", "To Kill a Mockingbird"]:
        try:
            src_idx   = model._get_book_idx(title)
            src_score = model.df_books.loc[src_idx, "complexity_score"]
            recs      = model.recommend(title, top_n=3)
            print(f"\n'{title}' (skor={src_score:.1f}) →")
            if recs:
                for i, r in enumerate(recs, 1):
                    print(f"  {i}. {r['title'][:45]:<45} "
                          f"[{r['language']}] "
                          f"skor={r['complexity_score']:.1f} "
                          f"delta={r['complexity_delta']:+.1f} "
                          f"sim={r['similarity_score']:.3f}")
            else:
                print("     (tidak ada rekomendasi — perlu dataset lebih besar)")
        except ValueError as e:
            print(f"  [SKIP] {e}")

    print("\n→ Lanjutkan ke step4_api.py")
    return model


if __name__ == "__main__":
    model = training_pipeline()