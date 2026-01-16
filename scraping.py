# ==========================================
# SCRAPING SCRIPT - YOUTUBE COMMENTS
# Output: dataset_youtube_unlabeled.csv
# ==========================================

import pandas as pd
from youtube_comment_downloader import *
import time

def npm_20241310055_scrape_youtube_comments():
    print("=== MEMULAI SCRAPING YOUTUBE (NPM: 20241310055) ===")
    
    # 1. Setup
    downloader = YoutubeCommentDownloader()
    video_urls = [
        'https://www.youtube.com/watch?v=KJdt-HBBGIo',
        'https://www.youtube.com/watch?v=J_tFEaOJdFU'
    ]
    
    all_data = []
    TARGET_PER_VIDEO = 10000 # Ambil SEMUA komentar (Max 10k)
    
    # 2. Loop URLs
    for url in video_urls:
        print(f"\\\\nProcessing: {url}")
        try:
            start_time = time.time()
            comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
            
            count = 0
            for comment in comments:
                text = comment['text']
                
                # Filter spam/pendek
                if len(text.split()) > 3:
                    all_data.append(text)
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  - Got {count} comments...")
                
                if count >= TARGET_PER_VIDEO:
                    break
            
            print(f"Selesai! Dapat {count} komentar dari video ini.")
            
        except Exception as e:
            print(f"Error pada {url}: {e}")
            
    # 3. Save to CSV
    df = pd.DataFrame(all_data, columns=['text'])
    
    # Bersihkan duplikat jika ada
    df.drop_duplicates(subset=['text'], inplace=True)
    
    filename = 'dataset_youtube_unlabeled.csv'
    df.to_csv(filename, index=False)
    
    print(f"\\\\n=== SCRAPING SELESAI ===")
    print(f"Total Data Unik: {len(df)}")
    print(f"Disimpan ke: {filename}")
    print("Langkah selanjutnya: Jalankan script labeling Llama 3.")

if __name__ == "__main__":
    npm_20241310055_scrape_youtube_comments()
