# ==========================================
# LABELING SCRIPT - LEXICON BASED (CEPAT)
# Input: dataset_youtube_unlabeled.csv
# Output: dataset_labeled_final.csv
# ==========================================

import pandas as pd
import sys
import subprocess

# Auto-install progress bar
try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

def npm_20241310055_lexicon_label(text):
    """
    Labeling berdasarkan kamus kata sederhana.
    Sangat cepat, tapi tidak seakurat Llama 3.
    """
    text_lower = str(text).lower()
    
    # Kamus Kata Positif (Diperkaya)
    pos_words = [
        'bagus', 'keren', 'mantap', 'setuju', 'hebat', 'pintar', 'cerdas', 
        'solutif', 'tegas', 'menang', 'lanjutkan', 'terbaik', 'dukung', 
        'wow', 'rapi', 'jelas', 'berani', 'paten', 'gemoy', 'semangat', 
        'juara', 'all in', '01', '02', '03', 'bismillah', 'alhamdulillah',
        'berwibawa', 'bijaksana', 'visioner', 'memuaskan', 'masuk akal', 
        'logis', 'data', 'fakta', 'santun', 'tenang', 'menguasai', 
        'pro rakyat', 'amanah', 'jujur', 'menyala', 'idola', 'panutan', 
        'respect', 'salut', 'bangga', 'sukses', 'menang telak', 'kualitas',
        'terbukti', 'nyata', 'sayang', 'cinta', 'love', 'gass', 'oke', 'sip',
        'top', 'jos', 'pasti', 'yakin', 'masuk', 'percaya', 'gas', 'sah', 
        'amin', 'aamiin', 'doa', 'semoga', 'merakyat', 'tulus', 'ikhlas'
    ]
    
    # Kamus Kata Negatif (Diperkaya + Toxic Words + Slang)
    neg_words = [
        'jelek', 'kecewa', 'bodoh', 'lemah', 'kalah', 'bohong', 'omong kosong', 
        'emosi', 'tidak setuju', 'hancur', 'mundur', 'takut', 'payah', 
        'gagal', 'bingung', 'licik', 'curang', 'malas', 'benci', 'muak',
        'nangis', 'bacot', 'tolol', 'dungu', 'hoax', 'kasar', 'tidak sopan',
        'omon', 'omon-omon', 'baper', 'blunder', 'ngawur', 'fitnah', 
        'drama', 'pencitraan', 'munafik', 'korup', 'dinasti', 'halu',
        'tidak jelas', 'muter-muter', 'omdo', 'janji palsu', 'parah', 'badut',
        'emosian', 'provokasi', 'menyerang', 'sombong', 'angkuh', 'zalim',
        'tai', 'anjing', 'bangsat', 'kampret', 'sialan', 'goblok', 'idiot', 
        'gila', 'mampus', 'sampah', 'setan', 'iblis', 'najis', 'jijik',
        'aneh', 'kacau', 'suram', 'hadeuh', 'gajelas', 'rosak', 'sebel',
        'rusak', 'ancur', 'hancur'
    ]
    
    score = 0
    for w in pos_words:
        if w in text_lower: score += 1
    for w in neg_words:
        if w in text_lower: score -= 1
        
    if score > 0: return 'positif'
    if score < 0: return 'negatif'
    return 'netral'

def npm_20241310055_run_labeling_simple():
    print("=== MEMULAI LABELING (METODE KAMUS/CEPAT) (NPM: 20241310055) ===")
    
    input_file = 'dataset_youtube_unlabeled.csv'
    output_file = 'dataset_labeled_final.csv'
    
    try:
        df = pd.read_csv(input_file)
        print(f"Data dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print(f"[ERROR] File '{input_file}' tidak ditemukan.")
        return

    print("Sedang melakukan labeling...")
    results = []
    
    for text in tqdm(df['text']):
        label = npm_20241310055_lexicon_label(text)
        results.append(label)
        
    df['label'] = results
    
    # Hapus Netral -> DIBATALKAN (User minta 3 Kelas: Positif, Negatif, Netral)
    # df = df[df['label'] != 'netral'] 
    
    # Simpan
    df.to_csv(output_file, index=False)
    print(f"\\n=== SELESAI ===")
    print(f"Data tersimpan di: {output_file}")
    print("Preview Distribusi Label:")
    print(df['label'].value_counts())
    print("\\n[INFO] Anda sekarang bisa lanjut ke script Soal 1.")

if __name__ == "__main__":
    npm_20241310055_run_labeling_simple()
