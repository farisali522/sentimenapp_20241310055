# =================================================================================================
# JAWABAN UAS DEEP LEARNING - SOAL 1 (SENTIMENT ANALYSIS)
# NAMA  : FARIS ALI HUSAMUDDIN
# NPM   : 20241310055
# =================================================================================================

import os
import subprocess
import sys

# --- BAGIAN 1: INSTALASI OTOMATIS (AGAR BISA JALAN DI COLAB/LOKAL) ---
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['pandas', 'numpy', 'seaborn', 'matplotlib', 'scikit-learn', 'nltk', 'PySastrawi', 'youtube-comment-downloader']
print("Mengecek dan menginstall library yang dibutuhkan...")
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        pkg_map = {'scikit-learn': 'sklearn', 'PySastrawi': 'Sastrawi'}
        install_name = package
        import_name = pkg_map.get(package, package)
        try:
            __import__(import_name)
        except ImportError:
            print(f"Menginstall {package}...")
            install_package(package)

# --- BAGIAN 2: IMPORT LIBRARY ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
from youtube_comment_downloader import *

# Download NLTK Data Lengkap (6 Poin)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- BAGIAN 3: FUNGSI UTAMA (PREFIX NPM: 20241310055) ---

def npm_20241310055_preprocessing(text):
    """
    PREPROCESSING LENGKAP (6 POIN):
    1. Normalization (Case Folding + Regex)
    2. Parsing (Sentence/Pattern)
    3. Tokenization (NLTK)
    4. Stopword Removal
    5. Lemmatization (WordNet)
    6. Stemming (Porter)
    """
    # 1. Normalization (Case Folding + Regex)
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)    # Hapus URL
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Hapus Mention
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus Angka & Simbol (Parsing Pattern)
    
    # Setup Stopwords
    try:
        stopwords_id = stopwords.words('indonesian')
    except:
        stopwords_id = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu']
    stopwords_en = stopwords.words('english')
    
    # 3. Tokenization (NLTK)
    words = nltk.word_tokenize(text)
    
    # 4. Stopword Removal
    words = [w for w in words if w not in stopwords_id and w not in stopwords_en]
    
    # 5. Lemmatization (Legacy form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # 6. Stemming (Skipped - User Request)
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)

def npm_20241310055_main_process():
    print(f"\n[PROSES 1] LOAD DATASET (NPM: 20241310055)")
    filename = 'dataset_labeled_final.csv'
    
    try:
        df = pd.read_csv(filename)
        print(f"Berhasil membaca {filename}")
        print(f"Total Data Awal: {len(df)}")
        
        # FILTER: HANYA POSITIF & NEGATIF (Binary Classification)
        df = df[df['label'].isin(['positif', 'negatif'])]
        print(f"Total Data Setelah Filter (Binary): {len(df)}")
        
    except FileNotFoundError:
        print(f"[ERROR] File '{filename}' tidak ditemukan.")
        return

    # 2. PREPROCESSING
    print(f"\n[PROSES 2] MELAKUKAN PREPROCESSING LENGKAP (6 POIN)...")
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(npm_20241310055_preprocessing)
    print("Preview Preprocessing:")
    print(df[['text', 'clean_text']].head())
    
    # [NEW] SAVE CLEAN DATA (To Speed up Streamlit)
    df.to_csv('dataset_labeled_final.csv', index=False)
    print(f"[INFO] Dataset dengan 'clean_text' berhasil disimpan ke 'dataset_labeled_final.csv'")

    # 3. MODELLING
    print(f"\n[PROSES 3] TRAINING MODEL LOGISTIC REGRESSION...")
    X = df['clean_text']
    y = df['label']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # 4. EVALUASI
    print(f"\n[PROSES 4] EVALUASI MODEL...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    
    # Save Model (PENTING UNTUK GITHUB)
    import pickle
    with open('model_uas_20241310055.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer_uas_20241310055.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"\n[INFO] Model & Vectorizer BARU (Support 6 Poin Preprocessing) Berhasil Disimpan!")

# --- EXECUTE ---
if __name__ == "__main__":
    npm_20241310055_main_process()
