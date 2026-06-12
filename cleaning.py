import pandas as pd
import re

file_path = "data/raw/gramedia_scraped.csv"

print("[INFO] Membaca file CSV...")
df = pd.read_csv(file_path)

def bersihkan_sisa_teks(teks):
    if pd.isna(teks):
        return teks
        
    teks = str(teks)
    # 1. Jika ada minimal 3 bintang, potong teks di posisi tersebut
    if "***" in teks:
        teks = teks.split("***")[0]
        
    # 2. Hapus sisa bintang, strip (-), atau spasi yang nyangkut di akhir kalimat
    teks = re.sub(r'[\*\-\s]+$', '', teks)
    return teks.strip()

print("[INFO] Membersihkan kolom sinopsis...")
df['summary'] = df['summary'].apply(bersihkan_sisa_teks)

# Simpan kembali
df.to_csv(file_path, index=False, encoding="utf-8-sig")
print("[SELESAI] Tanda bintang berhasil dibasmi dari dataset!")