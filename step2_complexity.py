"""
=================================================================
PETA JALAN LITERASI - STEP 2: HITUNG COMPLEXITY SCORE
=================================================================
Strategi (Opsi A):
  - Buku BAHASA INGGRIS  → LFTK (brucewlee/lftk) dengan spaCy en_core_web_sm
  - Buku BAHASA INDONESIA → Implementasi manual (textstat + nltk)
    karena LFTK dirancang untuk English dan tidak ada spaCy model ID

Fitur LFTK yang dipakai (dipilih berdasarkan korelasi tinggi
dengan readability dari paper BEA @ ACL 2023):
  Domain surface, family wordsent    : t_word, t_sent, t_uword
  Domain surface, family avgwordsent : a_word_ps, a_char_pw
  Domain surface, family typetokenratio : t_uword/t_word (TTR)
  Domain surface, family readformula : Flesch-Kincaid, Gunning Fog

Fitur manual untuk bahasa Indonesia (ekuivalen):
  Rata-rata panjang kalimat & kata
  Type-Token Ratio
  Proporsi kata panjang (7+ karakter)
  Rasio karakter per kata (a_char_pw)
  Estimasi suku kata via hitung vokal (a,i,u,e,o) — proxy morfologi ID
  Rasio suku kata per kata — pengganti Flesch/Gunning Fog yg berbasis morfologi EN

Skor Akhir (0–100):
  0–20  : Sangat Mudah
  21–40 : Mudah
  41–60 : Menengah
  61–80 : Sulit
  81–100: Sangat Sulit

Jalankan: python step2_complexity.py
=================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import re
import nltk
import textstat
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── PATH ───────────────────────────────────────────────────────
CLEAN_FILE  = os.path.join("data", "clean", "books_clean.csv")
SCORED_FILE = os.path.join("data", "clean", "books_scored.csv")
FEAT_FILE   = os.path.join("data", "clean", "books_features.csv")

# ─── SETUP ──────────────────────────────────────────────────────

def setup_nltk():
    for path, name in [
        ("tokenizers/punkt_tab",                    "punkt_tab"),
        ("corpora/stopwords",                        "stopwords"),
        ("taggers/averaged_perceptron_tagger_eng",   "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK] Downloading {name}...")
            nltk.download(name, quiet=True)


def load_spacy_and_lftk():
    """
    Load spaCy en_core_web_sm dan modul lftk.
    Return (nlp, lftk_module) atau (None, None) jika gagal.
    """
    try:
        import spacy
        import lftk
        try:
            nlp = spacy.load("en_core_web_sm")
            print("[OK] spaCy en_core_web_sm loaded")
        except OSError:
            print("[INFO] Mengunduh en_core_web_sm...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            print("[OK] spaCy en_core_web_sm loaded")
        return nlp, lftk
    except ImportError as e:
        print(f"[WARN] LFTK/spaCy tidak tersedia: {e}")
        print("       Buku EN akan pakai fallback manual.")
        return None, None


# ─── FEATURE KEYS LFTK YANG KITA PAKAI ─────────────────────────
# Dipilih dari family yang language="general" atau "en",
# relevan untuk readability, dan ringan di CPU.
#
# Referensi kode asli LFTK (brucewlee/lftk):
#   LFTK = lftk.Extractor(docs=doc)
#   LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)
#   features = LFTK.extract(features=LFTK_FEATURE_KEYS)
#
LFTK_FEATURE_KEYS = [
    # surface > wordsent (foundation)
    "t_word",       # total number of words
    "t_uword",      # total number of unique words
    "t_sent",       # total number of sentences
    "t_char",       # total number of characters
    # surface > avgwordsent (derivation)
    "a_word_ps",    # average words per sentence
    "a_char_pw",    # average characters per word
    # surface > typetokenratio
    "t_uword",      # dipakai ulang untuk TTR = t_uword / t_word
    # surface > readformula (English only)
    "a_kup_pw",     # average Kuperman Age-of-Acquisition per word
]

# Subset yang language="general" (aman untuk semua bahasa jika perlu)
LFTK_GENERAL_KEYS = [
    "t_word", "t_uword", "t_sent", "t_char",
    "a_word_ps", "a_char_pw",
]


# ─── EXTRACTOR BAHASA INGGRIS (LFTK) ────────────────────────────

class EnglishExtractor:
    """
    Ekstrak fitur linguistik untuk teks bahasa Inggris menggunakan LFTK.
    Cara pakai LFTK (dari source asli brucewlee/lftk):

      doc  = nlp(text)                          # spaCy doc
      LFTK = lftk.Extractor(docs=doc)           # inisialisasi
      LFTK.customize(stop_words=True,           # sertakan stopword
                     punctuations=False,         # exclude tanda baca
                     round_decimal=3)
      feat = LFTK.extract(features=keys)        # dict hasil ekstraksi
    """

    def __init__(self, nlp, lftk_module):
        self.nlp  = nlp
        self.lftk = lftk_module

        # Ambil semua feature key yang tersedia dari LFTK
        # (search_features adalah fungsi di lftk module langsung)
        try:
            all_keys = self.lftk.search_features(
                language="general", return_format="list_key"
            )
            # Tambah readformula (English-only, aman karena kita sudah di blok EN)
            read_keys = self.lftk.search_features(
                family="readformula", return_format="list_key"
            )
            self.feature_keys = list(dict.fromkeys(all_keys + read_keys))
            print(f"[LFTK] {len(self.feature_keys)} feature keys loaded")
        except Exception:
            self.feature_keys = LFTK_GENERAL_KEYS

    def extract(self, text: str) -> dict:
        if not isinstance(text, str) or len(text.strip()) < 20:
            return self._fallback(text)
        try:
            # Batasi panjang teks agar ringan di CPU (sinopsis biasanya <500 kata)
            doc = self.nlp(text[:3000])

            LFTK_extractor = self.lftk.Extractor(docs=doc)
            LFTK_extractor.customize(
                stop_words=True,      # hitung stopword
                punctuations=False,   # abaikan tanda baca
                round_decimal=3
            )
            features = LFTK_extractor.extract(features=self.feature_keys)

            # Hitung TTR manual dari hasil LFTK
            t_word  = features.get("t_word",  1) or 1
            t_uword = features.get("t_uword", 1) or 1
            features["ttr"] = round(t_uword / t_word, 3)

            return features

        except Exception as e:
            return self._fallback(text)

    def _fallback(self, text: str) -> dict:
        """Fallback ke manual jika LFTK gagal — pakai stopword EN, bukan ID."""
        return IndonesianExtractor._extract_manual(text, lang="en")


# ─── EXTRACTOR BAHASA INDONESIA (MANUAL) ────────────────────────

# Vokal bahasa Indonesia — dipakai untuk estimasi jumlah suku kata.
# Bahasa Indonesia bersifat aglutinatif dan pola suku katanya
# sangat teratur: umumnya (K)V atau (K)VK, sehingga menghitung
# vokal adalah proxy suku kata yang jauh lebih akurat daripada
# algoritma Flesch/Gunning Fog yang berbasis morfologi bahasa Inggris.
VOWELS_ID = set("aiueoAIUEO")

# Stopword bahasa Indonesia (manual, karena NLTK tidak punya korpus ID)
STOPWORDS_ID = {
    "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "dari",
    "ke", "ada", "tidak", "juga", "sudah", "akan", "ia", "mereka",
    "kita", "kamu", "kami", "pada", "oleh", "dalam", "sebuah",
    "satu", "dua", "tiga", "bisa", "harus", "adalah", "tersebut",
    "saat", "jika", "maka", "namun", "tetapi", "karena", "seperti",
    "bahwa", "atau", "nya", "pun", "pula", "lalu", "kemudian",
    "ketika", "setelah", "sebelum", "antara", "hingga", "sampai",
    "meski", "walaupun", "agar", "supaya", "melalui", "tentang",
}


def count_vowels_id(word: str) -> int:
    """
    Hitung jumlah vokal dalam satu kata sebagai estimasi jumlah suku kata.
    Contoh:
      "buku"        → 2 vokal (bu-ku)          → 2 suku kata
      "membaca"     → 3 vokal (mem-ba-ca)       → 3 suku kata
      "pembelajaran"→ 6 vokal (pem-be-la-jar-an)→ 5 suku kata (approx)
    Ini lebih valid untuk bahasa Indonesia daripada algoritma Flesch
    yang menghitung suku kata berdasarkan pola konsonan kluster EN.
    """
    return sum(1 for ch in word if ch in VOWELS_ID)


class IndonesianExtractor:
    """
    Ekstrak fitur linguistik untuk teks bahasa Indonesia secara manual.
    LFTK tidak digunakan karena tidak ada model spaCy khusus bahasa ID.

    Pengganti Flesch Reading Ease & Gunning Fog (keduanya berbasis
    morfologi bahasa Inggris):
      → avg_vowels_pw  : rata-rata vokal per kata (proxy suku kata ID)
      → vowel_ratio    : rasio vokal / total karakter (kepadatan fonetik)
      → syllable_complexity : rasio kata dengan ≥4 vokal (kata panjang/kompleks)

    Fitur lain tetap ekuivalen dengan output LFTK untuk EN.
    """

    def extract(self, text: str) -> dict:
        return self._extract_manual(text, lang="id")

    @staticmethod
    def _extract_manual(text: str, lang: str = "id") -> dict:
        """
        Ekstrak fitur linguistik secara manual.

        Args:
            text : teks sinopsis buku
            lang : "id" → stopword Indonesia
                   "en" → stopword Inggris dari NLTK
                   (dipakai saat dipanggil sebagai fallback EnglishExtractor)
        """
        if not isinstance(text, str) or len(text.strip()) < 20:
            return IndonesianExtractor._empty()

        # Pilih stopword sesuai bahasa
        if lang == "en":
            try:
                stop_words = set(nltk.corpus.stopwords.words("english"))
            except Exception:
                stop_words = set()
        else:
            stop_words = STOPWORDS_ID

        # Tokenisasi kalimat
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in text.split(".") if s.strip()]

        words = re.findall(r'\b[a-zA-Z\u00C0-\u024F]+\b', text.lower())

        if not words:
            return IndonesianExtractor._empty()

        t_word  = len(words)
        t_uword = len(set(words))
        t_sent  = max(len(sentences), 1)
        t_char  = sum(len(w) for w in words)

        # Ekuivalen a_word_ps (LFTK: average words per sentence)
        sent_word_counts = [
            len(re.findall(r'\b[a-zA-Z\u00C0-\u024F]+\b', s))
            for s in sentences
            if s.strip()
        ]
        a_word_ps = float(np.mean(sent_word_counts)) if sent_word_counts else float(t_word)

        # Ekuivalen a_char_pw (LFTK: average characters per word)
        a_char_pw = t_char / t_word if t_word else 0.0

        # TTR (ekuivalen t_uword/t_word dari LFTK)
        ttr = t_uword / t_word if t_word else 0.0

        # Stopword count — stopword disesuaikan dengan lang
        t_stopword = sum(1 for w in words if w in stop_words)

        # Proporsi kata panjang (7+ karakter) — proxy leksikal
        long_word_ratio = sum(1 for w in words if len(w) >= 7) / t_word

        # ── Fitur berbasis suku kata (khusus/lebih valid untuk ID) ──
        # Flesch & Gunning Fog TIDAK dipakai untuk ID karena algoritma
        # mereka menghitung suku kata via pola kluster konsonan EN.
        # Sebagai gantinya kita hitung vokal langsung dengan regex.

        vowel_counts = [count_vowels_id(w) for w in words]

        # Rata-rata vokal per kata → proxy "panjang suku kata"
        avg_vowels_pw = float(np.mean(vowel_counts)) if vowel_counts else 0.0

        # Rasio total vokal / total karakter → kepadatan fonetik
        total_vowels = sum(vowel_counts)
        vowel_ratio  = total_vowels / t_char if t_char else 0.0

        # Proporsi kata dengan ≥4 vokal → kata multi-suku-kata / kompleks
        syllable_complexity = sum(1 for v in vowel_counts if v >= 4) / t_word

        # ── TAMBAHAN: Flesch & Gunning Fog untuk Fallback EN ──
        # Gunakan textstat jika teksnya Inggris agar compute_score tidak error
        fre_score = 50.0
        gfi_score = 10.0
        if lang == "en":
            try:
                fre_score = textstat.flesch_reading_ease(text)
                gfi_score = textstat.gunning_fog(text)
            except Exception:
                pass

        return {
            # ── Fitur universal (nama key sejajar dengan output LFTK) ──
            "t_word":              t_word,
            "t_uword":             t_uword,
            "t_sent":              t_sent,
            "t_char":              t_char,
            "t_stopword":          t_stopword,
            "a_word_ps":           round(a_word_ps, 3),
            "a_char_pw":           round(a_char_pw, 3),
            "ttr":                 round(ttr, 3),
            "long_word_ratio":     round(long_word_ratio, 3),
            # ── Fitur suku kata (pengganti Flesch/Gunning Fog untuk ID) ──
            "avg_vowels_pw":       round(avg_vowels_pw, 3),
            "vowel_ratio":         round(vowel_ratio, 3),
            "syllable_complexity": round(syllable_complexity, 3),
            # ── Fitur Fallback EN (textstat) ──
            "fre":                 fre_score,
            "gfi":                 gfi_score,
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "t_word": 0, "t_uword": 0, "t_sent": 1, "t_char": 0,
            "t_stopword": 0, "a_word_ps": 10.0, "a_char_pw": 5.0,
            "ttr": 0.5, "long_word_ratio": 0.2,
            "avg_vowels_pw": 2.5, "vowel_ratio": 0.4,
            "syllable_complexity": 0.15,
            "fre": 50.0, "gfi": 10.0,
        }


# ─── COMPUTE COMPLEXITY SCORE (UNIFIED) ─────────────────────────

def compute_complexity_score(features: dict, lang: str) -> float:
    """
    Hitung skor kompleksitas 0–100 dari dict fitur.

    Komponen berbeda antara EN dan ID:

    BAHASA INGGRIS (fitur dari LFTK):
      1. Flesch Difficulty (fre, invert)    — 30%
      2. Avg words per sentence (a_word_ps) — 20%
      3. Avg chars per word (a_char_pw)     — 20%
      4. TTR (type-token ratio)             — 10%
      5. Gunning Fog (gfi)                  — 20%

    BAHASA INDONESIA (fitur manual berbasis morfologi ID):
      1. Syllable complexity (kata ≥4 vokal) — 30%  ← pengganti Flesch
      2. Avg words per sentence (a_word_ps)  — 20%
      3. Avg chars per word (a_char_pw)      — 20%
      4. TTR (type-token ratio)              — 10%
      5. Avg vowels per word (avg_vowels_pw) — 20%  ← pengganti Gunning Fog
    """

    def clip(val, lo, hi):
        return max(lo, min(hi, float(val or 0)))

    # ── Komponen 2: Avg words per sentence (sama untuk EN dan ID) ─
    a_word_ps = features.get("a_word_ps", 15.0)
    s_sent    = clip((a_word_ps - 5) / 35 * 100, 0, 100)

    # ── Komponen 3: Avg chars per word (sama untuk EN dan ID) ─────
    a_char_pw = features.get("a_char_pw", 5.0)
    s_word    = clip((a_char_pw - 3) / 9 * 100, 0, 100)

    # ── Komponen 4: TTR (sama untuk EN dan ID) ────────────────────
    ttr   = features.get("ttr", 0.5)
    s_ttr = clip((ttr - 0.2) / 0.7 * 100, 0, 100)

    if lang == "en":
        # ── EN Komponen 1: Flesch Difficulty (dari LFTK) ──────────
        # Key LFTK untuk Flesch Reading Ease: "fre"
        flesch_re = features.get("fre",
                    features.get("flesch_re",
                    features.get("flesch_reading_ease", 50.0)))
        flesch_re = clip(flesch_re, 0, 100)
        s_readability = 100 - flesch_re   # invert: rendah RE = sulit

        # ── EN Komponen 5: Gunning Fog (dari LFTK) ────────────────
        # Key LFTK untuk Gunning Fog Index: "gfi"
        fog   = features.get("gfi", features.get("gunning_fog", 10.0))
        s_fog = clip((fog - 5) / 20 * 100, 0, 100)

        score = (
            0.30 * s_readability +
            0.20 * s_sent        +
            0.20 * s_word        +
            0.10 * s_ttr         +
            0.20 * s_fog
        )

    else:
        # ── ID Komponen 1: Syllable Complexity ────────────────────
        # Proporsi kata dengan ≥4 vokal (kata multi-suku-kata/kompleks).
        # Range tipikal: 0.05 (teks sangat sederhana) – 0.40 (teks akademis)
        # Contoh tinggi: "pembelajaran", "keberhasilan", "perkembangan"
        syl_complex = features.get("syllable_complexity", 0.15)
        s_readability = clip(syl_complex / 0.40 * 100, 0, 100)

        # ── ID Komponen 5: Avg Vowels Per Word ────────────────────
        # Range tipikal bahasa Indonesia: 1.5 (kata pendek) – 4.5 (kata panjang)
        # Kata sederhana: "di" (1), "dan" (1), "buku" (2)
        # Kata kompleks:  "pembelajaran" (5), "perkembangan" (4)
        avg_vow = features.get("avg_vowels_pw", 2.5)
        s_fog   = clip((avg_vow - 1.5) / 3.0 * 100, 0, 100)

        score = (
            0.30 * s_readability +
            0.20 * s_sent        +
            0.20 * s_word        +
            0.10 * s_ttr         +
            0.20 * s_fog
        )

    return round(float(np.clip(score, 0, 100)), 2)


def get_complexity_label(score: float) -> str:
    if score <= 20:   return "Sangat Mudah"
    elif score <= 40: return "Mudah"
    elif score <= 60: return "Menengah"
    elif score <= 80: return "Sulit"
    else:             return "Sangat Sulit"


# ─── PIPELINE UTAMA ─────────────────────────────────────────────

def complexity_pipeline():
    print("\n" + "="*60)
    print("  PETA JALAN LITERASI - STEP 2: COMPLEXITY SCORING")
    print("  EN: LFTK (brucewlee/lftk) | ID: Manual (textstat+nltk)")
    print("="*60)

    if not os.path.exists(CLEAN_FILE):
        print(f"[ERROR] {CLEAN_FILE} tidak ditemukan.")
        print("        Jalankan step1_preprocess.py terlebih dahulu!")
        return None

    setup_nltk()
    df = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")
    print(f"[OK] Loaded {len(df)} buku")
    print(f"     Bahasa Indonesia : {(df['language']=='id').sum()}")
    print(f"     Bahasa Inggris   : {(df['language']=='en').sum()}")

    # Load LFTK + spaCy (hanya untuk buku EN)
    nlp, lftk_module = load_spacy_and_lftk()
    use_lftk = (nlp is not None and lftk_module is not None)

    en_extractor = EnglishExtractor(nlp, lftk_module) if use_lftk else None
    id_extractor = IndonesianExtractor()

    scores   = []
    labels   = []
    feat_recs = []
    en_count = id_count = lftk_count = fallback_count = 0

    print(f"\n[PROSES] Menghitung complexity score...")
    if use_lftk:
        print(f"         Buku EN  → LFTK (API asli brucewlee/lftk)")
    else:
        print(f"         Buku EN  → Manual fallback (LFTK tidak tersedia)")
    print(f"         Buku ID  → Manual (textstat + nltk)")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        text = str(row.get("summary", ""))
        lang = str(row.get("language", "en")).strip().lower()

        if lang == "en":
            en_count += 1
            if use_lftk and en_extractor:
                features = en_extractor.extract(text)
                lftk_count += 1
            else:
                features = IndonesianExtractor._extract_manual(text, lang="en")
                fallback_count += 1
        else:
            id_count += 1
            features = id_extractor.extract(text)

        score = compute_complexity_score(features, lang)
        label = get_complexity_label(score)

        scores.append(score)
        labels.append(label)
        feat_recs.append({
            "book_id":    row.get("book_id", 0),
            "language":   lang,
            **{k: v for k, v in features.items() if not isinstance(v, (dict, list))},
        })

    df["complexity_score"] = scores
    df["complexity_label"] = labels

    # Simpan
    df.to_csv(SCORED_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(feat_recs).to_csv(FEAT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n[SUKSES] {SCORED_FILE}")
    print(f"         {FEAT_FILE}")
    print(f"\n--- RINGKASAN PROSES ---")
    print(f"  Buku EN diproses    : {en_count}")
    if use_lftk:
        print(f"    └─ via LFTK       : {lftk_count}")
        print(f"    └─ via fallback   : {fallback_count}")
    print(f"  Buku ID diproses    : {id_count}")
    print(f"\n--- DISTRIBUSI SKOR ---")
    print(df["complexity_label"].value_counts().to_string())
    print(f"\n  Min  : {df['complexity_score'].min():.1f}")
    print(f"  Max  : {df['complexity_score'].max():.1f}")
    print(f"  Rata : {df['complexity_score'].mean():.1f}")
    print(f"\n--- CONTOH HASIL ---")
    sample = df[["title","language","complexity_score","complexity_label"]].head(6)
    print(sample.to_string(index=False))
    print("\n→ Lanjutkan ke step3_train_recommender.py")

    return df


if __name__ == "__main__":
    df = complexity_pipeline()