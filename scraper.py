"""
=================================================================
PETA JALAN LITERASI - SCRAPER GRAMEDIA (MASS SCRAPING ULTIMATE)
=================================================================
Fungsi: Mengambil 1000 data buku dari Gramedia.com.
Alur  : Kategori -> Klik "Muat Lebih Banyak" -> Ekstrak Buku
Output: data/raw/gramedia_scraped.csv
=================================================================
"""

import time
import json
import os
import re
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ─── KONFIGURASI ────────────────────────────────────────────────
TARGET_URL = "https://www.gramedia.com/categories/buku/fiksi-sastra"
MAX_BOOKS = 1000  # Target jumlah buku
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gramedia_scraped.csv")

def ensure_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def init_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return driver

def scrape_gramedia():
    ensure_dir()
    driver = init_driver()
    books_data = []
    master_book_urls = []
    
    print(f"\n[INFO] Membuka halaman kategori utama: {TARGET_URL}")
    driver.get(TARGET_URL)
    time.sleep(4)

    # ==============================================================
    # FASE 1: KUMPULKAN TAUTAN SUB-KATEGORI
    # ==============================================================
    print("[INFO] Mengumpulkan tautan sub-kategori...")
    subcategories = []
    try:
        subcat_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/categories/buku/fiksi-sastra/']")
        for elem in subcat_elements:
            url = elem.get_attribute("href")
            if url and url not in subcategories:
                subcategories.append(url)
    except Exception as e:
        print(f"[ERROR] Gagal mengambil sub-kategori: {e}")

    if not subcategories:
        subcategories = [TARGET_URL]
    else:
        print(f"[INFO] Ditemukan {len(subcategories)} sub-kategori.")

    # ==============================================================
    # FASE 2: KLIK "MUAT LEBIH BANYAK" & KUMPULKAN TAUTAN BUKU
    # ==============================================================
    for subcat_url in subcategories:
        if len(master_book_urls) >= MAX_BOOKS:
            break
            
        print(f"\n[INFO] Memasuki sub-kategori: {subcat_url.split('/')[-1]}")
        driver.get(subcat_url)
        time.sleep(3)
        
        print("       Mencari tombol 'Muat Lebih Banyak'...")
        for _ in range(20):
            # Scroll ke area bawah tempat tombol biasa berada
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 800);")
            time.sleep(2)
            
            try:
                # Cari khusus elemen <button> agar tidak salah klik teks lain
                btn_xpath = "//button[contains(translate(., 'MUAT LEBIH BANYAK', 'muat lebih banyak'), 'muat lebih banyak')]"
                load_more_btn = driver.find_element(By.XPATH, btn_xpath)
                
                # CEK 1: Apakah tombolnya benar-benar terlihat?
                if not load_more_btn.is_displayed():
                    print("       [-] Tombol sudah disembunyikan. Lanjut...")
                    break
                
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn)
                time.sleep(1)
                
                # CEK 2: Hitung buku sebelum klik
                books_before = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/products/']"))
                
                # Lakukan klik
                driver.execute_script("arguments[0].click();", load_more_btn)
                print("       [+] Klik 'Muat Lebih Banyak' dieksekusi.")
                time.sleep(3) # Tunggu buku baru termuat
                
                # Hitung buku setelah klik
                books_after = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/products/']"))
                if books_after == books_before:
                    print("       [-] Tidak ada buku baru yang muncul. Lanjut...")
                    break # Keluar dari loop klik
                    
            except Exception:
                # Jika tombol benar-benar tidak ditemukan di HTML
                break
                
        # Kumpulkan semua link buku yang berhasil dimuat
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/products/']")
        added_count = 0
        for link in links:
            url = link.get_attribute("href")
            if url and url not in master_book_urls:
                master_book_urls.append(url)
                added_count += 1
                if len(master_book_urls) >= MAX_BOOKS:
                    break
        print(f"       Mendapatkan {added_count} tautan buku baru. (Total sementara: {len(master_book_urls)})")

    print(f"\n[INFO] Selesai mengumpulkan tautan. Total target eksekusi: {len(master_book_urls)} buku.\n")

    # ==============================================================
    # FASE 3: EKSTRAKSI DATA BUKU VIA JSON-LD
    # ==============================================================
    try:
        for idx, url in enumerate(master_book_urls):
            print(f"[{idx+1}/{len(master_book_urls)}] Membuka: {url}")
            driver.get(url)
            time.sleep(2)
            
            try:
                title = "Anonim"
                author = "Anonim"
                summary = ""
                raw_genres = []
                language_code = "id"

                # 1. Ambil Genre dari Visual
                try:
                    breadcrumb_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/categories/']")
                    for link in breadcrumb_links:
                        cat_text = link.text.strip()
                        if cat_text and cat_text.lower() not in ["home", "buku"]:
                            if cat_text not in raw_genres:
                                raw_genres.append(cat_text)
                except:
                    pass

                # 2. Ambil Data dari JSON-LD
                ld_scripts = driver.find_elements(By.XPATH, "//script[@type='application/ld+json']")
                for script in ld_scripts:
                    try:
                        data = json.loads(script.get_attribute("innerHTML"))
                        if isinstance(data, dict): data = [data]
                            
                        for item in data:
                            if item.get("@type") == "DataFeed" and "dataFeedElement" in item:
                                for feed_item in item["dataFeedElement"]:
                                    if feed_item.get("@type") == "Book":
                                        title = feed_item.get("name", title)
                                        summary = feed_item.get("description", summary)
                                        
                                        lang_raw = feed_item.get("inLanguage", "")
                                        if lang_raw and "english" in lang_raw.lower():
                                            language_code = "en"
                                            
                                        if "author" in feed_item and isinstance(feed_item["author"], list):
                                            author = feed_item["author"][0].get("name", author)
                    except:
                        continue

                # Fallback Judul
                if title == "Anonim":
                    try:
                        title = driver.find_element(By.CSS_SELECTOR, "meta[property='og:title']").get_attribute("content").split("|")[0].strip()
                    except: pass

                # 3. Cleansing Genre
                cleaned_genres = []
                for g in raw_genres:
                    g = g.replace('Fiksi', 'Fiction').replace('fiksi', 'fiction').capitalize()
                    if g not in cleaned_genres:
                        cleaned_genres.append(g)
                genre_str = "|".join(cleaned_genres[-2:]) if cleaned_genres else "Fiction"
                
                # 4. Cleansing Sinopsis
                if summary:
                    summary = re.sub(r'\s+', ' ', summary).strip()
                    stop_phrases = [
                        "Selling Point:", "Selling point:", "Profil Penulis:",
                        "Tentang Penulis", "Pernahkah Anda terpikir", 
                        "Tahun Terbit", "*******************"
                    ]
                    cut_index = len(summary)
                    for phrase in stop_phrases:
                        idx_pos = summary.find(phrase)
                        if idx_pos != -1 and idx_pos < cut_index:
                            cut_index = idx_pos
                    summary = summary[:cut_index].strip()

                if summary and len(summary) > 50 and title != "Anonim":
                    books_data.append({
                        "title": title,
                        "author": author,
                        "genre": genre_str,
                        "language": language_code,
                        "summary": summary,
                        "complexity_score": np.nan
                    })
                    print(f"  [SUKSES] {title} | Lang: {language_code}")
                else:
                    print(f"  [SKIP] Data tidak lengkap.")
                    
            except Exception as e:
                print(f"  [GAGAL] Terjadi error: {e}")
                continue

    finally:
        # ==============================================================
        # FASE 4: SIMPAN KE CSV
        # ==============================================================
        driver.quit()
        if books_data:
            df = pd.DataFrame(books_data)
            df.insert(0, "book_id", range(5001, 5001 + len(df)))
            df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"\n[SELESAI] {len(df)} Buku berhasil disimpan ke: {OUTPUT_FILE}")
        else:
            print("\n[INFO] Tidak ada data buku yang berhasil diekstrak.")

if __name__ == "__main__":
    scrape_gramedia()