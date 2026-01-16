# 🗳️ Analisis Sentimen Debat Capres 2024 (UAS Deep Learning)

**Nama:** FARIS ALI HUSAMUDDIN
**NPM:** 20241310055  
**Mata Kuliah:** Deep Learning  

---

## 📋 Deskripsi Proyek
Aplikasi ini adalah tugas UAS untuk melakukan **Analisis Sentimen** terhadap komentar YouTube mengenai Debat Capres 2024.
Aplikasi dibangun menggunakan **Streamlit** dan mencakup **5 Tahapan Utama** sesuai soal ujian:

1.  **Input Data**: Memuat dataset komentar YouTube (sudah dilabeli: Positif/Negatif/Netral).
2.  **Preprocessing**: Pembersihan teks (Case Folding, URL removal), Stopword Removal (ID & EN), dan **English Stemming**.
3.  **Modeling**: Pelatihan model menggunakan algoritma **Logistic Regression**.
4.  **Evaluation**: Pengujian akurasi model & visualisasi Confusion Matrix.
5.  **Prediction**: Tes prediksi sentimen pada kalimat baru.

## 🚀 Cara Menjalankan (Lokal)

1.  **Clone Repository**
    ```bash
    git clone https://github.com/farisali522/sentimentapp_20241310055.git
    cd sentimentapp_20241310055
    ```

2.  **Install Library**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi**
    ```bash
    streamlit run soal_2_sentiment_app_20241310055.py
    ```

## 🌐 Deployment
Aplikasi ini di-deploy menggunakan **Streamlit Cloud**.
Link Demo: *(Silakan isi link Streamlit Cloud Anda di sini setelah deploy)*

---
*Dibuat dengan ❤️ untuk UAS Deep Learning.*
