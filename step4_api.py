"""
=================================================================
PETA JALAN LITERASI - STEP 4: REST API (Flask)
=================================================================
Endpoint yang tersedia:

  GET  /api/health
       → Status server + jumlah buku dalam dataset

  GET  /api/books/search?q=<query>&lang=<id|en|all>
       → Cari buku berdasarkan judul/penulis

  GET  /api/books/<book_id>
       → Detail satu buku berdasarkan ID

  POST /api/recommend
       Body JSON: { "book_title": "Laskar Pelangi", "top_n": 5 }
       → Rekomendasi untuk satu buku

  POST /api/roadmap
       Body JSON: {
         "read_books": ["Laskar Pelangi", "Perahu Kertas"],
         "top_n": 5
       }
       → Peta jalan lengkap dari daftar buku yang sudah dibaca

─────────────────────────────────────────────────────────────────
KONTRAK JSON RESPONSE (untuk integrasi frontend):
Semua response sukses menggunakan format:
  {
    "status": "success",
    "data": { ... }       ← Payload utama
  }
Semua error menggunakan format:
  {
    "status": "error",
    "message": "..."
  }
─────────────────────────────────────────────────────────────────

Jalankan: python step4_api.py
Akses di : http://localhost:5000
=================================================================
"""

import os
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# WAJIB: import class model agar pickle bisa menemukan definisinya saat load.
# Tanpa ini pickle.load() akan throw AttributeError: Can't get attribute
# 'LiteracyRoadmapRecommender' karena class-nya tidak dikenal di namespace ini.
from step3_train_recommender import LiteracyRoadmapRecommender, get_complexity_label

# ─── KONFIGURASI ────────────────────────────────────────────────
MODEL_FILE = os.path.join("data", "model", "recommender.pkl")
PORT       = 5000
DEBUG      = True    # Set False untuk production

# ─── INISIALISASI FLASK ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)    # Izinkan request dari frontend (cross-origin)

# ─── LOAD MODEL ─────────────────────────────────────────────────
model = None

def load_model():
    """Load model dari file pickle."""
    global model
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_FILE}")
        print("        Jalankan step3_train_recommender.py terlebih dahulu!")
        return False
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"[OK] Model loaded. Dataset: {len(model.df_books)} buku")
    return True


# ─── HELPER ─────────────────────────────────────────────────────

def success_response(data: dict, status_code: int = 200):
    """Format response sukses yang konsisten."""
    return jsonify({"status": "success", "data": data}), status_code


def error_response(message: str, status_code: int = 400):
    """Format response error yang konsisten."""
    return jsonify({"status": "error", "message": message}), status_code


def check_model():
    """Pastikan model sudah dimuat sebelum handle request."""
    if model is None:
        return error_response("Model belum dimuat. Jalankan training terlebih dahulu.", 503)
    return None


# ─── ENDPOINT: HEALTH CHECK ─────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """
    GET /api/health

    Response JSON:
    {
      "status": "success",
      "data": {
        "service":      (str)  Nama layanan
        "model_loaded": (bool) Apakah model tersedia
        "total_books":  (int)  Jumlah buku dalam dataset
        "genres":       (list) Daftar genre yang tersedia
        "languages":    (list) ["id", "en"]
      }
    }
    """
    if model is None:
        return success_response({
            "service":      "Peta Jalan Literasi API",
            "model_loaded": False,
            "total_books":  0,
            "genres":       [],
            "languages":    [],
        })

    # Kumpulkan genre unik
    all_genres = set()
    for g_str in model.df_books["genre"].fillna(""):
        for g in g_str.split("|"):
            g = g.strip()
            if g:
                all_genres.add(g)

    return success_response({
        "service":      "Peta Jalan Literasi API",
        "model_loaded": True,
        "total_books":  len(model.df_books),
        "genres":       sorted(list(all_genres))[:50],   # Batasi 50 genre teratas
        "languages":    ["id", "en"],
    })


# ─── ENDPOINT: CARI BUKU ────────────────────────────────────────

@app.route("/api/books/search", methods=["GET"])
def search_books():
    """
    GET /api/books/search?q=<query>&lang=<id|en|all>&limit=<int>

    Query params:
      q     (str, required) : Kata kunci judul atau penulis
      lang  (str, optional) : Filter bahasa "id", "en", atau "all" (default)
      limit (int, optional) : Jumlah hasil maksimal (default: 10)

    Response JSON:
    {
      "status": "success",
      "data": {
        "query":       (str)  Query yang digunakan
        "total_found": (int)  Jumlah hasil
        "books": [
          {
            "book_id":          (int)   ID unik buku
            "title":            (str)   Judul buku
            "author":           (str)   Nama penulis
            "genre":            (str)   Genre dengan separator "|"
            "genre_list":       (list)  Genre dalam bentuk array
            "language":         (str)   "id" atau "en"
            "summary_short":    (str)   150 karakter pertama sinopsis
            "complexity_score": (float) Skor 0-100
            "complexity_label": (str)   "Mudah", "Menengah", dll
            "complexity_delta": null    (null untuk search result)
            "similarity_score": null    (null untuk search result)
            "roadmap_step":     null    (null untuk search result)
          },
          ...
        ]
      }
    }
    """
    err = check_model()
    if err:
        return err

    q     = request.args.get("q", "").strip()
    lang  = request.args.get("lang", "all").strip().lower()
    limit = int(request.args.get("limit", 10))

    if not q:
        return error_response("Parameter 'q' (query) wajib diisi.")

    df = model.df_books.copy()

    # Filter bahasa
    if lang in ("id", "en"):
        df = df[df["language"] == lang]

    # Cari di judul dan penulis (case-insensitive)
    mask = (
        df["title"].str.lower().str.contains(q.lower(), na=False) |
        df["author"].str.lower().str.contains(q.lower(), na=False)
    )
    results = df[mask].head(limit)

    books = [model._format_book(row) for _, row in results.iterrows()]

    return success_response({
        "query":       q,
        "total_found": len(books),
        "books":       books,
    })


# ─── ENDPOINT: DETAIL BUKU ──────────────────────────────────────

@app.route("/api/books/<int:book_id>", methods=["GET"])
def get_book(book_id: int):
    """
    GET /api/books/<book_id>

    Berbeda dari endpoint search: endpoint ini mengembalikan
    sinopsis PENUH (bukan summary_short 150 karakter) karena
    dipakai untuk halaman/modal detail buku yang memang
    dimaksudkan agar pengguna bisa membaca seluruh sinopsis.

    Response JSON:
    {
      "status": "success",
      "data": {
        "book": {
          ...semua field _format_book()...
          "summary_full": (str)  ← sinopsis lengkap tanpa dipotong
        }
      }
    }
    """
    err = check_model()
    if err:
        return err

    matches = model.df_books[model.df_books["book_id"] == book_id]
    if len(matches) == 0:
        return error_response(f"Buku dengan ID {book_id} tidak ditemukan.", 404)

    row  = matches.iloc[0]
    book = model._format_book(row)

    # Tambahkan sinopsis penuh — _format_book() hanya simpan 150 karakter
    # di summary_short untuk efisiensi response search/roadmap.
    # Untuk detail view, kita kirim teks lengkapnya.
    book["summary_full"] = str(row.get("summary", "")).strip()

    return success_response({"book": book})


# ─── ENDPOINT: REKOMENDASI SATU BUKU ────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    POST /api/recommend

    Request Body JSON:
    {
      "book_title": (str, required) Judul buku yang sudah dibaca
      "top_n":      (int, optional) Jumlah rekomendasi, default 5
    }

    Response JSON:
    {
      "status": "success",
      "data": {
        "source_book": {
          "book_id":          (int)
          "title":            (str)
          "author":           (str)
          "genre":            (str)
          "genre_list":       (list)
          "language":         (str)
          "summary_short":    (str)
          "complexity_score": (float)
          "complexity_label": (str)
          "complexity_delta": null
          "similarity_score": null
          "roadmap_step":     null
        },
        "recommendations": [
          {
            "book_id":          (int)
            "title":            (str)
            "author":           (str)
            "genre":            (str)
            "genre_list":       (list)   ← Digunakan untuk render tag di UI
            "language":         (str)    ← "id"/"en" untuk badge bahasa
            "summary_short":    (str)    ← Teks preview kartu buku
            "complexity_score": (float)  ← Tampilkan sebagai progress bar
            "complexity_label": (str)    ← Badge warna berdasarkan label
            "complexity_delta": (float)  ← "+7.3" → tampilkan sebagai "+7 poin"
            "similarity_score": (float)  ← 0–1, untuk sorting/display
            "roadmap_step":     (str)    ← Teks deskriptif langkah peta jalan
          },
          ...
        ],
        "next_target_range": {
          "min": (float)   ← Skor minimum zona target
          "max": (float)   ← Skor maksimum zona target
        }
      }
    }
    """
    err = check_model()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    book_title = body.get("book_title", "").strip()
    top_n      = int(body.get("top_n", 5))

    if not book_title:
        return error_response("Field 'book_title' wajib diisi.")

    if top_n < 1 or top_n > 20:
        return error_response("top_n harus antara 1 sampai 20.")

    try:
        source_idx  = model._get_book_idx(book_title)
        source_row  = model.df_books.loc[source_idx]
        source_book = model._format_book(source_row)
        source_score = float(source_row["complexity_score"])

        recommendations = model.recommend(book_title, top_n=top_n)

        return success_response({
            "source_book":       source_book,
            "recommendations":   recommendations,
            "next_target_range": {
                "min": round(source_score + 3, 1),
                "max": round(source_score + 15, 1),
            },
        })

    except ValueError as e:
        return error_response(str(e), 404)
    except Exception as e:
        return error_response(f"Error internal: {str(e)}", 500)


# ─── ENDPOINT: PETA JALAN LENGKAP ───────────────────────────────

@app.route("/api/roadmap", methods=["POST"])
def roadmap():
    """
    POST /api/roadmap

    Request Body JSON:
    {
      "read_books": (list, required) Daftar judul buku yang sudah dibaca
                    Contoh: ["Laskar Pelangi", "Perahu Kertas"]
      "top_n":      (int, optional)  Jumlah rekomendasi, default 5
    }

    Response JSON:
    {
      "status": "success",
      "data": {
        "current_level":  (float)  Rata-rata complexity buku yang dibaca
        "current_label":  (str)    Label level saat ini
        "target_range": {
          "min": (float),           ← Batas bawah zona target
          "max": (float)            ← Batas atas zona target
        },
        "source_books": [           ← Buku yang sudah dibaca (verified)
          { ... }                   ← Format _format_book()
        ],
        "recommendations": [        ← Rekomendasi buku berikutnya
          {
            "book_id":          (int)
            "title":            (str)
            "author":           (str)
            "genre":            (str)
            "genre_list":       (list)   ← Render sebagai chip/tag di UI
            "language":         (str)    ← Badge "ID" atau "EN"
            "summary_short":    (str)    ← Teks preview kartu buku
            "complexity_score": (float)  ← Progress bar 0–100
            "complexity_label": (str)    ← Warna badge sesuai label
            "complexity_delta": (float)  ← Tampilkan "+X poin"
            "similarity_score": (float)  ← Tidak perlu ditampilkan ke user
            "roadmap_step":     (str)    ← Teks tooltip/label langkah
          },
          ...
        ],

        // ── DATA VISUAL UNTUK "PETA PERJALANAN" ──────────────────
        // Gunakan data ini untuk menggambar visualisasi roadmap di frontend
        "roadmap_visualization": {
          "nodes": [                ← Setiap node = satu buku di peta
            {
              "node_id":     (int)    ID unik untuk React key
              "book_id":     (int)    Referensi ke buku
              "title":       (str)    Label node
              "score":       (float)  Posisi di sumbu kompleksitas
              "label":       (str)    Label kategori
              "type":        (str)    "read" | "recommended"
              "step_number": (int)    Urutan di peta (1, 2, 3, ...)
            },
            ...
          ],
          "edges": [                ← Koneksi antar node
            {
              "from_node_id": (int)
              "to_node_id":   (int)
              "delta":        (float) Selisih complexity
            },
            ...
          ],
          "complexity_axis": {      ← Untuk render sumbu Y di frontend
            "min":   0,
            "max":   100,
            "zones": [
              {"label": "Sangat Mudah", "min": 0,  "max": 20,  "color": "#4CAF50"},
              {"label": "Mudah",        "min": 20, "max": 40,  "color": "#8BC34A"},
              {"label": "Menengah",     "min": 40, "max": 60,  "color": "#FFC107"},
              {"label": "Sulit",        "min": 60, "max": 80,  "color": "#FF9800"},
              {"label": "Sangat Sulit", "min": 80, "max": 100, "color": "#F44336"}
            ]
          }
        }
      }
    }
    """
    err = check_model()
    if err:
        return err

    body       = request.get_json(silent=True) or {}
    read_books = body.get("read_books", [])
    top_n      = int(body.get("top_n", 5))

    if not read_books or not isinstance(read_books, list):
        return error_response("Field 'read_books' harus berupa array judul buku.")

    if len(read_books) > 20:
        return error_response("Maksimal 20 buku per request.")

    try:
        result = model.get_roadmap(read_books, top_n=top_n)

        if "error" in result:
            return error_response(result["error"], 404)

        # ── Bangun data visualisasi untuk frontend ──────────────
        nodes = []
        edges = []
        node_counter = 1

        # Node: buku yang sudah dibaca
        read_node_ids = []
        for book in result["source_books"]:
            node = {
                "node_id":     node_counter,
                "book_id":     book["book_id"],
                "title":       book["title"],
                "score":       book["complexity_score"],
                "label":       book["complexity_label"],
                "type":        "read",
                "step_number": node_counter,
            }
            nodes.append(node)
            read_node_ids.append(node_counter)
            node_counter += 1

        # Node: buku rekomendasi
        # Edge dihubungkan dari SEMUA buku yang sudah dibaca ke setiap rekomendasi,
        # bukan hanya dari buku terakhir. Ini yang membuat peta jalan
        # merepresentasikan seluruh profil bacaan pengguna, bukan satu buku saja.
        for i, rec in enumerate(result["recommendations"]):
            node = {
                "node_id":     node_counter,
                "book_id":     rec["book_id"],
                "title":       rec["title"],
                "score":       rec["complexity_score"],
                "label":       rec["complexity_label"],
                "type":        "recommended",
                "step_number": len(result["source_books"]) + i + 1,
            }
            nodes.append(node)

            # Sambungkan dari SETIAP buku yang dibaca ke rekomendasi ini
            for read_node_id in read_node_ids:
                edges.append({
                    "from_node_id": read_node_id,
                    "to_node_id":   node_counter,
                    "delta":        rec["complexity_delta"],
                })

            node_counter += 1

        # Zona kompleksitas untuk sumbu visualisasi
        complexity_zones = [
            {"label": "Sangat Mudah", "min": 0,  "max": 20,  "color": "#4CAF50"},
            {"label": "Mudah",        "min": 20, "max": 40,  "color": "#8BC34A"},
            {"label": "Menengah",     "min": 40, "max": 60,  "color": "#FFC107"},
            {"label": "Sulit",        "min": 60, "max": 80,  "color": "#FF9800"},
            {"label": "Sangat Sulit", "min": 80, "max": 100, "color": "#F44336"},
        ]

        result["roadmap_visualization"] = {
            "nodes":            nodes,
            "edges":            edges,
            "complexity_axis":  {
                "min":   0,
                "max":   100,
                "zones": complexity_zones,
            },
        }

        return success_response(result)

    except Exception as e:
        return error_response(f"Error internal: {str(e)}", 500)


# ─── ERROR HANDLERS ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return error_response("Endpoint tidak ditemukan.", 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return error_response("Method tidak diizinkan.", 405)

@app.errorhandler(500)
def internal_error(e):
    return error_response("Error internal server.", 500)


# ─── ENTRY POINT ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PETA JALAN LITERASI - STEP 4: API SERVER")
    print("="*55)

    if not load_model():
        print("\n[INFO] Server berjalan tanpa model.")
        print("       Jalankan step1-3 terlebih dahulu, lalu restart.")
    else:
        print(f"[OK] API siap di http://localhost:{PORT}")
        print("\nEndpoint tersedia:")
        print(f"  GET  http://localhost:{PORT}/api/health")
        print(f"  GET  http://localhost:{PORT}/api/books/search?q=laskar")
        print(f"  GET  http://localhost:{PORT}/api/books/1")
        print(f"  POST http://localhost:{PORT}/api/recommend")
        print(f"  POST http://localhost:{PORT}/api/roadmap")

    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)