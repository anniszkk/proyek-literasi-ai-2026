"""
=================================================================
PETA JALAN LITERASI - STEP 1: PRA-PEMROSESAN DATA
=================================================================
Fungsi:
  1. Download dataset CMU (buku bahasa Inggris, ~16k buku)
  2. Load & gabungkan CSV buku Indonesia dari scraping Gramedia
     (data/raw/gramedia_scraped.csv) jika tersedia
  3. Bersihkan teks, parse genre, deteksi bahasa
  4. Simpan ke data/clean/books_clean.csv

Sumber data:
  - CMU Book Summary Dataset → buku EN
  - data/raw/gramedia_scraped.csv → buku ID dari Gramedia
  - generate_indonesian_books() → 18 buku ID kurasi manual (fallback)

Jalankan: python step1_preprocess.py
=================================================================
"""

import os
import re
import ast
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── KONFIGURASI ────────────────────────────────────────────────
DATA_DIR   = "data"
RAW_DIR    = os.path.join(DATA_DIR, "raw")
CLEAN_DIR  = os.path.join(DATA_DIR, "clean")

DATASET_URL    = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
RAW_FILE       = os.path.join(RAW_DIR, "booksummaries.tar.gz")
GRAMEDIA_FILE  = os.path.join(RAW_DIR, "gramedia_scraped.csv")   # ← CSV Gramedia
CLEAN_FILE     = os.path.join(CLEAN_DIR, "books_clean.csv")

MAX_EN_BOOKS   = 4800   # Batas buku EN dari CMU agar total tidak terlalu besar

COLUMNS_OUT = [
    "book_id", "title", "author", "genre",
    "language", "summary", "complexity_score",
]

# ─── FREEBASE ID → NAMA GENRE ───────────────────────────────────
# CMU dataset menyimpan genre sebagai Freebase ID (misal /m/02xlf).
# Tabel ini memetakan ID yang paling sering muncul ke nama yang
# manusia bisa baca. ID yang tidak ada di tabel ini akan dibuang.
FREEBASE_GENRE_MAP = {
    "/m/02xlf":  "Fiction",
    "/m/05hgj":  "Novel",
    "/m/06nbt":  "Adventure",
    "/m/017fp":  "Romance",
    "/m/06n90":  "Science Fiction",
    "/m/0hn10":  "Non-fiction",
    "/m/03npn":  "Mystery",
    "/m/03g3w":  "Children's literature",
    "/m/02js9":  "Historical novel",
    "/m/01hmnh":  "Short story",
    "/m/06wkf":  "Crime Fiction",
    "/m/0mz2":   "Horror",
    "/m/02p0szs": "Thriller",
    "/m/0dwly":  "Fantasy",
    "/m/07m5w1": "Graphic novel",
    "/m/04nh4":  "Poetry",
    "/m/0gf28":  "Young adult",
    "/m/04g_wd": "Dystopian",
    "/m/039vk":  "Biography",
    "/m/0hfjk":  "Autobiography",
    "/m/01jfsb":  "Literary fiction",
    "/m/03ff00": "Speculative fiction",
    "/m/016lj8": "Historical fiction",
    "/m/02n4kr": "Satire",
    "/m/03mfnf": "Spy fiction",
    "/m/0hwxm":  "Political fiction",
    "/m/02yq81": "Psychological fiction",
    "/m/06mq7":  "Philosophical fiction",
    "/m/06gtzk": "Epistolary novel",
    "/m/08sdrw": "Erotic fiction",
    "/m/0d6gr":  "Paranormal romance",
    "/m/098tmk": "Urban fantasy",
    "/m/08w0_f": "Hard science fiction",
    "/m/016475": "Social science fiction",
    "/m/012jgz": "Cyberpunk",
    "/m/0lsxr":  "Steampunk",
    "/m/01j1n2": "Space opera",
    "/m/014dfn": "Military science fiction",
    "/m/026llv5":"New Weird",
    "/m/028v3":  "Absurdist fiction",
    "/m/02ql9":  "Magical realism",
    "/m/035qb4": "Gothic fiction",
    "/m/05qgc":  "Suspense",
    "/m/0707q":  "Chick lit",
    "/m/017rf8": "Family saga",
    "/m/0127jb": "Bildungsroman",
    "/m/017ssy": "Metafiction",
    "/m/0hc1z":  "Postmodern literature",
    "/m/01j1n2": "Noir fiction",
    "/m/0xdf":   "Wuxia",
}


# ─── HELPER FUNCTIONS ───────────────────────────────────────────

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, CLEAN_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"[OK] Direktori siap: {DATA_DIR}/")


def download_dataset() -> bool:
    if os.path.exists(RAW_FILE):
        print(f"[SKIP] Dataset CMU sudah ada: {RAW_FILE}")
        return True
    print("[DOWNLOAD] Mengunduh CMU Book Summary Dataset...")
    try:
        resp = requests.get(DATASET_URL, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(RAW_FILE, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="booksummaries"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"[OK] Download selesai: {RAW_FILE}")
        return True
    except Exception as e:
        print(f"[WARN] Download gagal: {e}")
        return False


def load_cmu_dataset() -> pd.DataFrame | None:
    """
    Parse CMU Book Summary (tab-separated di dalam tar.gz).
    Format: WikiID \\t FreebaseID \\t Title \\t Author \\t Date \\t Genres \\t Summary
    """
    import tarfile
    records = []
    try:
        with tarfile.open(RAW_FILE, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8", errors="replace")
                        for line in content.strip().split("\n"):
                            parts = line.split("\t")
                            if len(parts) >= 7:
                                records.append(parts)
        df = pd.DataFrame(records, columns=[
            "wiki_id", "freebase_id", "title", "author",
            "pub_date", "genres_raw", "summary"
        ])
        print(f"[OK] Loaded {len(df)} buku dari CMU dataset")
        return df
    except Exception as e:
        print(f"[WARN] Gagal parse CMU dataset: {e}")
        return None


def parse_genres(raw_genres_str: str) -> str:
    """
    Parse genre dari format CMU dataset ke string bersih dipisah '|'.

    CMU menyimpan genre sebagai Python dict literal dengan Freebase ID sebagai key:
      {'/m/02xlf': 'Fiction', '/m/0hn10': 'Novel'}

    Masalah lama: json.loads() gagal karena single-quote, lalu fallback regex
    menangkap Freebase ID (/m/02xlf) bukan nama genre-nya.

    Solusi berlapis:
      1. ast.literal_eval() → parse dict Python natively (handle single-quote)
         → ambil VALUES (nama genre), buang KEYS (Freebase ID)
         → jika value masih berupa Freebase ID, lookup ke FREEBASE_GENRE_MAP
      2. Fallback regex → ambil teks di antara tanda kutip,
         filter yang berpola /m/xxxxx dan panjang < 3 karakter
    """
    if not isinstance(raw_genres_str, str) or not raw_genres_str.strip():
        return "Lainnya"

    # ── Coba ast.literal_eval ────────────────────────────────────
    try:
        parsed = ast.literal_eval(raw_genres_str)
        if isinstance(parsed, dict):
            genres = []
            for k, v in parsed.items():
                v = str(v).strip()
                k = str(k).strip()

                if v.startswith("/m/"):
                    # Value adalah Freebase ID → cari di lookup table
                    mapped = FREEBASE_GENRE_MAP.get(v, "")
                    if mapped:
                        genres.append(mapped)
                    # Jika tidak ada di tabel, buang (jangan tampilkan /m/xxx)
                elif len(v) > 2:
                    genres.append(v)

            if genres:
                return "|".join(genres[:3])
    except Exception:
        pass

    # ── Fallback: regex ──────────────────────────────────────────
    # Tangkap semua string di antara tanda kutip tunggal atau ganda
    matches = re.findall(r"['\"]([^'\"]{3,})['\"]", raw_genres_str)
    genres  = [
        m for m in matches
        if not m.startswith("/m/")       # Buang Freebase ID
        and not re.match(r'^/\w', m)     # Buang path-like string
    ]
    return "|".join(genres[:3]) if genres else "Lainnya"


def clean_text(text: str) -> str:
    """Normalisasi whitespace dan hapus karakter aneh."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?\'\-]', ' ', text)
    return text.strip()


def detect_language_simple(text: str) -> str:
    """
    Deteksi bahasa dua lapis berdasarkan kata penanda Indonesia.

    Lapis 1 — Kata EKSKLUSIF Indonesia (satu saja → langsung "id"):
      Hampir tidak mungkin muncul di teks bahasa Inggris murni.

    Lapis 2 — Kata UMUM Indonesia (butuh ≥4 berbeda → "id"):
      Bisa muncul kebetulan di teks EN (nama tempat, dll).
    """
    if not isinstance(text, str):
        return "en"

    tokens = set(text.lower().split())

    id_exclusive = {
        "nya", "pun", "juga", "sudah", "belum", "agar", "namun",
        "tetapi", "supaya", "sedangkan", "meskipun", "walaupun",
        "bahwa", "karena", "ketika", "setelah", "sebelum", "sehingga",
    }
    if tokens & id_exclusive:
        return "id"

    id_common = {
        "yang", "dan", "di", "ini", "itu", "dengan", "untuk",
        "dari", "ke", "ada", "tidak", "akan", "ia", "mereka",
        "kita", "kamu", "kami", "pada", "oleh", "dalam", "adalah",
    }
    return "id" if len(tokens & id_common) >= 4 else "en"


# ─── DATASET BUKU INDONESIA (KURASI MANUAL) ─────────────────────

def generate_indonesian_books() -> pd.DataFrame:
    """
    18 buku Indonesia yang dikurasi manual sebagai fallback
    jika gramedia_scraped.csv tidak tersedia.
    Tersebar di 4 tingkat kompleksitas agar heuristik +5 poin
    bisa bekerja dalam satu genre.
    """
    books = [
        # Anak / Remaja
        {"title":"Diary Si Bocah Tengil","author":"Jeff Kinney",
         "genre":"Anak|Humor|Fiksi","language":"id",
         "summary":"Greg Heffley menulis buku harian tentang kehidupan sekolah menengahnya yang penuh kekacauan. Ia berusaha menjadi populer di antara teman-temannya. Kisah ringan dan lucu yang sangat digemari pembaca muda Indonesia."},
        {"title":"Hafalan Shalat Delisa","author":"Tere Liye",
         "genre":"Fiksi|Inspirasi|Religi","language":"id",
         "summary":"Delisa, gadis kecil di Aceh, menghafal shalat sebagai hadiah ulang tahun untuk ibunya. Tsunami 2004 merenggut keluarganya. Kisah tentang keteguhan iman, kesabaran, dan cinta seorang anak yang menghadapi bencana terbesar dalam hidupnya."},
        {"title":"Si Anak Kampoeng","author":"Masri Sareb Putra",
         "genre":"Anak|Inspirasi|Biografi","language":"id",
         "summary":"Kisah masa kecil seorang anak kampung di pedalaman Kalimantan yang berjuang mendapatkan pendidikan. Dengan semangat pantang menyerah, ia melewati berbagai rintangan untuk meraih cita-cita dan membuktikan bahwa asal usul bukan penghalang kesuksesan."},
        # Fiksi populer
        {"title":"Laskar Pelangi","author":"Andrea Hirata",
         "genre":"Fiksi|Drama|Inspirasi","language":"id",
         "summary":"Kisah sepuluh anak kampung di Belitung yang berjuang mendapatkan pendidikan di sekolah SD Muhammadiyah yang hampir roboh. Ikal dan sahabat-sahabatnya memiliki mimpi besar meski hidup dalam keterbatasan. Guru Muslimah menjadi inspirasi utama yang membentuk karakter mereka."},
        {"title":"Negeri 5 Menara","author":"Ahmad Fuadi",
         "genre":"Fiksi|Inspirasi|Drama","language":"id",
         "summary":"Alif meninggalkan Minangkabau untuk belajar di Pondok Pesantren Madani. Bersama lima sahabat dari berbagai penjuru Indonesia, ia bermimpi tentang dunia yang lebih luas. Moto man jadda wajada menjadi semangat mereka untuk meraih mimpi setinggi menara dunia."},
        {"title":"Perahu Kertas","author":"Dee Lestari",
         "genre":"Romansa|Fiksi|Drama","language":"id",
         "summary":"Kugy dan Keenan bertemu di Bandung dan terhubung melalui kreativitas dan mimpi bersama. Kugy seorang penulis cerita dongeng yang unik, Keenan seorang pelukis berbakat. Perjalanan cinta mereka diuji oleh jarak, waktu, dan tuntutan dunia nyata."},
        {"title":"Sang Pemimpi","author":"Andrea Hirata",
         "genre":"Fiksi|Inspirasi|Drama","language":"id",
         "summary":"Kelanjutan kisah Laskar Pelangi. Ikal dan Arai bermimpi kuliah di Universitas Sorbonne Paris. Mereka bekerja keras sebagai kuli ikan di pasar untuk membiayai sekolah. Persahabatan, cinta, dan tekad menjadi tema sentral cerita remaja yang mengharukan ini."},
        {"title":"Hujan","author":"Tere Liye",
         "genre":"Fiksi Ilmiah|Romansa|Drama","language":"id",
         "summary":"Di masa depan, teknologi memungkinkan manusia menghapus kenangan menyakitkan. Lail harus memilih antara melupakan Esok atau menyimpan semua kenangan pedih bersamanya. Sebuah kisah tentang cinta, kehilangan, dan keberanian menghadapi masa lalu."},
        {"title":"Dilan: Dia adalah Dilanku Tahun 1990","author":"Pidi Baiq",
         "genre":"Romansa|Fiksi|Drama","language":"id",
         "summary":"Milea, siswi pindahan dari Jakarta, bertemu Dilan di Bandung tahun 1990. Dilan, anggota geng motor yang cerdas dan jenaka, mendekatinya dengan cara-cara unik yang tak terlupakan. Kisah cinta remaja yang hangat dan penuh humor khas anak muda era 90-an."},
        # Fiksi menengah
        {"title":"Ronggeng Dukuh Paruk","author":"Ahmad Tohari",
         "genre":"Fiksi Sejarah|Drama|Sastra","language":"id",
         "summary":"Srintil menjadi ronggeng di dukuh terpencil dan mencintai Rasus dengan segenap jiwa. Kisah ini berlatar peristiwa G30S 1965, menggambarkan bagaimana masyarakat desa sederhana terseret arus politik besar yang tidak mereka pahami. Novel peraih berbagai penghargaan sastra bergengsi Indonesia."},
        {"title":"Pulang","author":"Leila S. Chudori",
         "genre":"Fiksi Sejarah|Drama|Sastra","language":"id",
         "summary":"Dimas Suryo terjebak di Paris saat G30S 1965 meletus dan tidak bisa pulang ke Indonesia. Anaknya, Lintang, tumbuh di Prancis sambil mencoba memahami identitas dan masa lalu ayahnya. Novel yang menggambarkan diaspora, trauma sejarah, dan kerinduan akan tanah air."},
        {"title":"Supernova: Ksatria, Puteri, dan Bintang Jatuh","author":"Dee Lestari",
         "genre":"Fiksi Ilmiah|Romansa|Sastra","language":"id",
         "summary":"Dimas dan Reuben menulis novel tentang ksatria dan puteri sebagai proyek rahasia mereka. Kisah fiksi yang mereka tulis secara misterius paralel dengan kehidupan nyata Ferre dan Rana di Jakarta. Novel ini memadukan fisika kuantum, spiritualitas, dan romansa dalam narasi berlapis yang inovatif."},
        {"title":"Tenggelamnya Kapal Van der Wijck","author":"Hamka",
         "genre":"Romansa|Drama|Sastra","language":"id",
         "summary":"Zainuddin, pemuda Makassar berdarah Minang, jatuh cinta pada Hayati di ranah Minang. Namun perbedaan adat dan status sosial memisahkan mereka. Kisah cinta tragis yang berlatar adat Minangkabau dan kritik terhadap feodalisme, ditulis dengan bahasa puitis yang kaya oleh ulama besar Hamka."},
        # Sastra / Sejarah
        {"title":"Bumi Manusia","author":"Pramoedya Ananta Toer",
         "genre":"Fiksi Sejarah|Drama|Sastra","language":"id",
         "summary":"Novel pertama dari Tetralogi Buru karya Pramoedya Ananta Toer. Minke, pribumi terpelajar di era kolonial Belanda akhir abad 19, jatuh cinta pada Annelies Mellema. Kisah ini menggambarkan pertentangan kelas, kolonialisme, dan kebangkitan kesadaran nasional di Hindia Belanda awal abad 20 dengan narasi yang kuat dan penuh kritik sosial tajam."},
        {"title":"Cantik Itu Luka","author":"Eka Kurniawan",
         "genre":"Fiksi Magis|Drama|Sastra","language":"id",
         "summary":"Dewi Ayu, perempuan cantik yang dipaksa menjadi pelacur di era kolonial, mengutuk anaknya yang keempat agar lahir buruk rupa. Novel ini menggunakan realisme magis untuk menceritakan luka sejarah Indonesia mulai era penjajahan hingga Orde Baru dengan teknik narasi yang kompleks dan simbolisme yang kaya makna."},
        {"title":"Anak Semua Bangsa","author":"Pramoedya Ananta Toer",
         "genre":"Fiksi Sejarah|Drama|Sastra","language":"id",
         "summary":"Novel kedua Tetralogi Buru. Minke memperluas pandangan dunianya dengan mempelajari sejarah perjuangan bangsa-bangsa lain melawan kolonialisme Eropa. Ia mulai memahami bahwa perjuangan melawan penindasan bukan hanya milik satu bangsa, tetapi merupakan perjuangan universal seluruh umat manusia yang terjajah."},
        # Non-fiksi / Akademis
        {"title":"Sejarah Tuhan","author":"Karen Armstrong",
         "genre":"Non-fiksi|Sejarah|Filsafat","language":"id",
         "summary":"Armstrong menelusuri sejarah konsep Tuhan dalam tiga agama Abrahamik selama empat ribu tahun. Dengan analisis komparatif yang mendalam, buku ini mengkaji bagaimana pemahaman manusia tentang Yang Ilahi berevolusi merespons konteks sejarah, budaya, dan kebutuhan spiritual masing-masing zaman."},
        {"title":"Negara Paripurna","author":"Yudi Latif",
         "genre":"Non-fiksi|Politik|Akademis","language":"id",
         "summary":"Kajian mendalam tentang historisitas, rasionalitas, dan aktualitas Pancasila sebagai dasar negara Indonesia. Latif menelusuri akar intelektual dan historis setiap sila, menganalisis konteks perdebatan para founding fathers, serta menawarkan tafsir kontemporer yang relevan untuk Indonesia masa kini."},
    ]
    df = pd.DataFrame(books)
    df.insert(0, "book_id", range(1, len(df) + 1))
    df["complexity_score"] = np.nan
    print(f"[OK] Dataset Indonesia kurasi manual: {len(df)} buku")
    return df


def load_gramedia_scraped() -> pd.DataFrame | None:
    """
    Load CSV hasil scraping Gramedia dari data/raw/gramedia_scraped.csv.
    Validasi dan normalisasi kolom agar sesuai format pipeline.
    """
    if not os.path.exists(GRAMEDIA_FILE):
        print(f"[INFO] gramedia_scraped.csv tidak ditemukan di {GRAMEDIA_FILE}")
        print("       Menggunakan dataset kurasi manual sebagai fallback.")
        return None

    try:
        df = pd.read_csv(GRAMEDIA_FILE, encoding="utf-8-sig")
        print(f"[OK] Loaded {len(df)} buku dari gramedia_scraped.csv")

        # Pastikan kolom wajib ada
        required = {"title", "summary"}
        missing  = required - set(df.columns)
        if missing:
            print(f"[WARN] Kolom wajib tidak ada: {missing} — skip file ini")
            return None

        # Isi kolom yang mungkin tidak ada dengan nilai default
        if "author"   not in df.columns: df["author"]   = ""
        if "genre"    not in df.columns: df["genre"]    = "Fiksi"
        if "language" not in df.columns: df["language"] = "id"

        # Bersihkan teks
        df["title"]   = df["title"].apply(clean_text)
        df["author"]  = df["author"].apply(clean_text)
        df["summary"] = df["summary"].apply(clean_text)

        # Buang baris dengan sinopsis terlalu pendek
        before = len(df)
        df = df[df["summary"].str.len() > 50].copy()
        print(f"[OK] Setelah filter sinopsis pendek: {len(df)} buku "
              f"(dibuang {before - len(df)})")

        df["complexity_score"] = np.nan
        return df

    except Exception as e:
        print(f"[WARN] Gagal baca gramedia_scraped.csv: {e}")
        return None


# ─── DEDUPLICATION ───────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus buku duplikat berdasarkan judul yang dinormalisasi.
    Normalisasi: lowercase, hapus tanda baca, strip whitespace.
    Prioritas: pertahankan baris pertama (buku Indonesia didahulukan
    karena df_id di-concat lebih dulu).
    """
    def norm(title: str) -> str:
        if not isinstance(title, str):
            return ""
        return re.sub(r'[^\w\s]', '', title.lower()).strip()

    df = df.copy()
    df["_norm_title"] = df["title"].apply(norm)
    before = len(df)
    df = df.drop_duplicates(subset=["_norm_title"], keep="first")
    df = df.drop(columns=["_norm_title"])
    removed = before - len(df)
    if removed:
        print(f"[DEDUP] Hapus {removed} duplikat judul")
    return df


# ─── PIPELINE UTAMA ─────────────────────────────────────────────

def preprocess_pipeline() -> pd.DataFrame:
    ensure_dirs()
    print("\n" + "="*58)
    print("  PETA JALAN LITERASI - STEP 1: PRA-PEMROSESAN DATA")
    print("="*58)

    # ── 1. Dataset Buku Indonesia ────────────────────────────────
    # Prioritas: gramedia_scraped.csv > kurasi manual
    print("\n[BAGIAN 1] Memuat dataset buku Indonesia...")
    df_id = load_gramedia_scraped()
    if df_id is None:
        df_id = generate_indonesian_books()

    # Pastikan kolom language terisi "id"
    df_id["language"] = df_id.get("language", pd.Series(["id"] * len(df_id)))
    df_id["language"] = df_id["language"].fillna("id")

    print(f"           Total buku Indonesia: {len(df_id)}")

    # ── 2. Dataset Buku Inggris (CMU) ───────────────────────────
    print("\n[BAGIAN 2] Memuat dataset buku Inggris (CMU)...")
    download_ok = download_dataset()
    df_en       = None

    if download_ok:
        df_raw = load_cmu_dataset()
        if df_raw is not None and len(df_raw) > 0:
            print("[PROSES] Membersihkan dan memformat dataset CMU...")

            df_en = pd.DataFrame()
            df_en["title"]   = df_raw["title"].apply(clean_text)
            df_en["author"]  = df_raw["author"].apply(clean_text)
            df_en["summary"] = df_raw["summary"].apply(clean_text)

            # Parse genre — ini yang fix masalah /m/xxxxx
            print("[PROSES] Parsing genre CMU (fix Freebase ID)...")
            df_en["genre"] = df_raw["genres_raw"].apply(parse_genres)

            # Deteksi bahasa dari sinopsis
            print("[PROSES] Mendeteksi bahasa buku CMU...")
            tqdm.pandas(desc="Deteksi Bahasa")
            df_en["language"] = df_en["summary"].progress_apply(detect_language_simple)

            # Filter hanya English (buku ID sudah dari Gramedia/kurasi)
            df_en = df_en[df_en["language"] == "en"]
            df_en = df_en[df_en["summary"].str.len() > 50]
            df_en["complexity_score"] = np.nan

            # Buang buku yang genrenya masih "Lainnya" DAN sinopsisnya pendek
            # (biasanya entri sampah dari CMU)
            df_en = df_en[
                ~((df_en["genre"] == "Lainnya") & (df_en["summary"].str.len() < 100))
            ]

            # Sampel agar tidak terlalu besar
            df_en = df_en.sample(
                n=min(MAX_EN_BOOKS, len(df_en)), random_state=42
            ).reset_index(drop=True)

            print(f"           Total buku Inggris: {len(df_en)}")

    if df_en is None or len(df_en) == 0:
        print("[WARN] Tidak ada data EN dari CMU — hanya pakai data ID")
        df_en = pd.DataFrame(columns=list(df_id.columns))

    # ── 3. Gabungkan ID + EN ────────────────────────────────────
    print("\n[BAGIAN 3] Menggabungkan dataset ID + EN...")

    # Samakan kolom sebelum concat
    for col in COLUMNS_OUT:
        if col not in df_id.columns: df_id[col] = np.nan
        if col not in df_en.columns: df_en[col] = np.nan

    # ID diletakkan di atas agar duplikat yang dipertahankan adalah versi ID
    df_merged = pd.concat(
        [df_id[COLUMNS_OUT], df_en[COLUMNS_OUT]],
        ignore_index=True
    )

    # ── 4. Deduplikasi & Reset book_id ──────────────────────────
    df_merged = deduplicate(df_merged)
    df_merged["book_id"]          = range(1, len(df_merged) + 1)
    df_merged["complexity_score"] = np.nan

    # ── 5. Validasi genre — buang sisa /m/xxxxx yang lolos ──────
    # Double-check: filter baris yang genrenya masih mengandung /m/
    before = len(df_merged)
    mask_bad_genre = df_merged["genre"].str.contains(r'/m/', na=False)
    if mask_bad_genre.any():
        # Coba remap dulu
        df_merged.loc[mask_bad_genre, "genre"] = (
            df_merged.loc[mask_bad_genre, "genre"]
            .apply(lambda g: "|".join(
                FREEBASE_GENRE_MAP.get(part.strip(), "")
                for part in g.split("|")
                if FREEBASE_GENRE_MAP.get(part.strip(), "")
            ) or "Lainnya")
        )
        still_bad = df_merged["genre"].str.contains(r'/m/', na=False)
        df_merged.loc[still_bad, "genre"] = "Lainnya"
        print(f"[FIX] Genre /m/xxx diperbaiki: {mask_bad_genre.sum()} buku")

    # ── 6. Simpan ───────────────────────────────────────────────
    df_final = df_merged[COLUMNS_OUT]
    df_final.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{'='*58}")
    print(f"  [SUKSES] {CLEAN_FILE}")
    print(f"  Total buku       : {len(df_final)}")
    print(f"  Bahasa Indonesia : {(df_final['language']=='id').sum()}")
    print(f"  Bahasa Inggris   : {(df_final['language']=='en').sum()}")
    n_lainnya = (df_final['genre'] == 'Lainnya').sum()
    n_freebase = df_final['genre'].str.contains(r'/m/', na=False).sum()
    print(f"  Genre 'Lainnya'  : {n_lainnya} buku")
    print(f"  Genre /m/xxx     : {n_freebase} buku (seharusnya 0)")
    print(f"{'='*58}")
    print("\n→ Lanjutkan ke step2_complexity.py")
    return df_final


# ─── ENTRY POINT ────────────────────────────────────────────────
if __name__ == "__main__":
    df = preprocess_pipeline()
    print("\nContoh output:")
    print(df[["title", "author", "genre", "language"]].head(5).to_string(index=False))