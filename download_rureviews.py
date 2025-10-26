#!/usr/bin/env python3
"""
Скачивание RuReviews датасета
"""

import os
import pandas as pd
import requests

def download_rureviews():
    """Скачивание RuReviews датасета"""
    print("Скачивание RuReviews...")
    
    # Создаем папку
    os.makedirs('data/rureviews', exist_ok=True)
    
    try:
        # Если файл уже есть локально
        source_file = "women-clothing-accessories.3-class.balanced.csv"
        
        if os.path.exists(source_file):
            # Копируем локальный файл
            df = pd.read_csv(source_file)
            df.to_csv('data/rureviews/reviews.csv', index=False)
            print(f"RuReviews скопирован из локального файла")
            
        else:
            # Пробуем скачать (если есть прямая ссылка)
            print("Локальный файл не найден")
            print("Вам нужно:")
            print("   1. Найти файл 'women-clothing-accessories.3-class.balanced.csv'")
            print("   2. Сохранить его в 'data/rureviews/reviews.csv'")
            
        # Анализируем данные
        analyze_rureviews()
        
    except Exception as e:
        print(f"Ошибка: {e}")

def analyze_rureviews():
    """Анализ структуры RuReviews"""
    print("\nАнализ RuReviews...")
    
    try:
        df = pd.read_csv('data/rureviews/reviews.csv')
        
        print(f"Данные загружены: {len(df)} строк")
        print(f"Колонки: {df.columns.tolist()}")
        
        # Покажем распределение классов
        if 'label' in df.columns:
            print(f"Распределение меток:")
            print(df['label'].value_counts().sort_index())
        elif 'rating' in df.columns:
            print(f"Распределение рейтингов:")
            print(df['rating'].value_counts().sort_index())
        
        # Покажем примеры
        print(f"Примеры текстов:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            text_preview = row.get('text', row.get('review', 'N/A'))[:100] + "..."
            label = row.get('label', row.get('rating', 'N/A'))
            print(f"   {i+1}. [{label}] {text_preview}")
            
    except Exception as e:
        print(f"Ошибка анализа: {e}")

if __name__ == "__main__":
    download_rureviews()