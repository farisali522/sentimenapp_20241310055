# ==========================================
# LABELING SCRIPT - LLAMA 3 (OLLAMA)
# Input: dataset_youtube_unlabeled.csv
# Output: dataset_labeled_final.csv
# ==========================================

import pandas as pd
import requests
import json
import time
import sys
import subprocess

# Auto-install progress bar (tqdm) biar keren
try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

def npm_20241310055_classify_with_ollama(text, model="llama3"):
    """
    Mengirim prompt ke Ollama Local untuk klasifikasi sentimen.
    """
    url = "http://localhost:11434/api/generate"
    
    # Prompt Engineering yang ketat agar outputnya konsisten
    prompt = f"""
    Analisis sentimen dari komentar berikut tentang Debat Capres Indonesia 2024.
    
    Konteks:
    - POSITIF: Mendukung salah satu paslon (01 Anies/02 Prabowo/03 Ganjar), memuji penampilan debat, atau optimisme Indonesia.
    - NEGATIF: Mengkritik/menghina salah satu paslon, kecewa dengan debat, hate speech, atau keluhan.
    - NETRAL: Informasi datar, pertanyaan tanpa emosi, atau komentar tidak relevan (spam).

    Komentar: "{text}"
    
    Hanya berikan jawaban satu kata: "POSITIF", "NEGATIF", atau "NETRAL".
    Jangan berikan penjelasan apapun.
    Jawaban:
    """
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0 # Agar deterministik
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()['response'].strip().upper()
            # Bersihkan jika ada tanda baca
            result = result.replace('.', '').replace('*', '')
            
            # Mapping ke lowercase standar
            if 'POSITIF' in result: return 'positif'
            if 'NEGATIF' in result: return 'negatif'
            if 'NETRAL' in result: return 'netral'
            return 'netral' # Default fallback
        else:
            print(f"Error Ollama: {response.status_code}")
            return 'netral'
    except Exception as e:
        print(f"Connection Error (Pastikan Ollama jalan): {e}")
        return 'error'

def npm_20241310055_run_labeling():
    print("=== MEMULAI LABELING DENGAN LLAMA 3 (NPM: 20241310055) ===")
    
    # 1. Load Data
    input_file = 'dataset_youtube_unlabeled.csv'
    output_file = 'dataset_labeled_final.csv'
    
    try:
        df = pd.read_csv(input_file)
        print(f"Data dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print(f"[ERROR] File '{input_file}' tidak ditemukan. Jalankan scraping.py dulu!")
        return

    # 2. Setup Ollama Check
    try:
        # Cek sekilas apakah ollama nyala
        requests.get("http://localhost:11434")
    except:
        print("[CRITICAL ERROR] Ollama tidak terdeteksi berjalan di port 11434.")
        print("Pastikan Anda sudah menjalankan aplikasi Ollama.")
        return

    # 3. Labeling Loop
    print("Sedang melakukan labeling (ini memakan waktu tergantung GPU)...")
    results = []
    
    # Gunakan tqdm untuk progress bar
    for text in tqdm(df['text']):
        label = npm_20241310055_classify_with_ollama(text)
        results.append(label)
        
    df['label'] = results
    
    # 4. Filter Error
    df = df[df['label'] != 'error']
    
    # 5. Save
    df.to_csv(output_file, index=False)
    print(f"\\\\n=== LABELING SELESAI ===")
    print(f"Data tersimpan di: {output_file}")
    print("Preview Distribusi Label:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    npm_20241310055_run_labeling()
