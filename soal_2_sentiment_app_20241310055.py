import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- CONFIGURATION ---
st.set_page_config(
    page_title="UAS Deep Learning - 20241310055",
    page_icon="🎓",
    layout="wide"
)

# --- LOAD RESOURCES (NPM: 20241310055) ---
@st.cache_resource
def npm_20241310055_load_data():
    try:
        df = pd.read_csv('dataset_labeled_final.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def npm_20241310055_load_model():
    try:
        with open('model_uas_20241310055.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer_uas_20241310055.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Download NLTK Support
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)      # Untuk Tokenization
nltk.download('punkt_tab', quiet=True)  # *** WAJIB UNTUK STREAMLIT CLOUD (FIX ERROR) ***
nltk.download('wordnet', quiet=True)    # Untuk Lemmatization
nltk.download('omw-1.4', quiet=True)

# --- PREPROCESSING FUNCTION ---
def npm_20241310055_preprocessing_lengkap(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        stopwords_id = stopwords.words('indonesian')
    except:
        stopwords_id = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu']
    stopwords_en = stopwords.words('english')
    
    # 3. Tokenization (Menggunakan NLTK)
    # Parsing sentence into words
    words = nltk.word_tokenize(text)
    
    # 4. Stopword Removal
    words = [w for w in words if w not in stopwords_id and w not in stopwords_en]
    
    # 5. Lemmatization (Mengembalikan ke bentuk dasar kamus)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # 6. Stemming (Skipped - User Request: English Stemming destroys Indo words)
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)

# --- MAIN APPLICATION ---
def main():
    st.title("🎓 UAS Deep Learning: Sentiment Analysis")
    st.markdown("**Nama:** FARIS ALI HUSAMUDDIN | **NPM:** 20241310055")
    
    # Load Data & Model
    df = npm_20241310055_load_data()
    model, vectorizer = npm_20241310055_load_model()
    
    if df is not None:
        # FILTER: HANYA POSITIF & NEGATIF (Binary)
        df = df[df['label'].isin(['positif', 'negatif'])]
    
    if df is None or model is None:
        st.error("File data/model tidak ditemukan. Pastikan file .csv dan .pkl ada.")
        return

    # --- TABS (3 MENU) ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset", 
        "📈 Model Evaluation", 
        "🤖 Prediksi Sentimen"
    ])
    
    # === TAB 1: DATASET ===
    with tab1:
        st.header("Dataset Komentar YouTube")
        st.write(f"Total Data: {len(df)} baris")
        
        st.info("""
        **Sumber Data (Scraping):**
        1. Debat Capres 2024 (Video 1): [https://www.youtube.com/watch?v=KJdt-HBBGIo](https://www.youtube.com/watch?v=KJdt-HBBGIo)
        2. Debat Capres 2024 (Video 2): [https://www.youtube.com/watch?v=J_tFEaOJdFU](https://www.youtube.com/watch?v=J_tFEaOJdFU)
        """)
        
        with st.expander("ℹ️ Metodologi Labeling (Llama 3)"):
            st.markdown("""
            Dataset ini dilabeli menggunakan **Large Language Model (Llama 3)** via Ollama.
            **Prompt yang digunakan:**
            ```text
            Analisis sentimen dari komentar berikut tentang Debat Capres Indonesia 2024.
            
            Konteks:
            - POSITIF: Mendukung paslon 01/02/03, memuji, optimisme.
            - NEGATIF: Mengkritik, hate speech, keluhan, kata tidak pantas.
            
            Hanya berikan jawaban satu kata: "POSITIF" atau "NEGATIF".
            ```
            """)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            st.write("Distribusi Label:")
            st.bar_chart(df['label'].value_counts())
        
    # === TAB 2: MODEL & EVALUATION ===
    with tab2:
        st.header("Evaluasi Model (Logistic Regression)")
        
        if st.button("Jalankan Evaluasi"):
            if 'clean_text' not in df.columns:
                 df['clean_text'] = df['text'].apply(npm_20241310055_preprocessing_lengkap)
            
            X = df['clean_text']
            y = df['label']
            X_vec = vectorizer.transform(X)
            y_pred = model.predict(X_vec)
            
            acc = accuracy_score(y, y_pred)
            st.metric("Akurasi Model", f"{acc*100:.2f}%")
            
            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
                st.pyplot(fig)
            
            with col_eval2:
                st.write("Detail Report:")
                st.text(classification_report(y, y_pred))

    # === TAB 3: PREDICTION ===
    with tab3:
        st.header("Uji Coba Model")
        
        input_pred = st.text_area("Masukkan Kalimat:", height=100, placeholder="Tulis komentar di sini...")
        
        if st.button("Analisis Sentimen"):
            if input_pred:
                # 1. Tampilkan Preprocessing
                clean_pred = npm_20241310055_preprocessing_lengkap(input_pred)
                
                # 2. Prediksi (Pure Model)
                vec_pred = vectorizer.transform([clean_pred])
                res = model.predict(vec_pred)[0]
                proba = model.predict_proba(vec_pred)[0]
                
                # 3. Hasil
                st.divider()
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Hasil Prediksi")
                    if res == 'positif':
                        st.success(f"### {res.upper()} 😊")
                    else:
                        st.error(f"### {res.upper()} 😠")
                        
                    with st.expander("Lihat Hasil Preprocessing (Pembersihan Teks)"):
                        st.code(clean_pred)
                
                with c2:
                    st.subheader("Confidence")
                    prob_df = pd.DataFrame({'Label': model.classes_, 'Probabilitas': proba})
                    st.bar_chart(prob_df.set_index('Label'))

if __name__ == '__main__':
    main()
