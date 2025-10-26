#!/usr/bin/env python3
"""
Исправленный скрипт для скачивания RuReviews
"""

import requests
import pandas as pd
import os

def download_rureviews():
    """Скачивание и преобразование RuReviews"""
    print("Скачивание RuReviews...")
    
    # Создаем папку
    os.makedirs('data/rureviews', exist_ok=True)
    
    # Скачиваем файл
    url = 'https://raw.githubusercontent.com/sismetanin/rureviews/master/women-clothing-accessories.3-class.balanced.csv'
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Ошибка скачивания: {response.status_code}")
        return
    
    # Сохраняем сырой файл
    with open('data/rureviews/raw_reviews.csv', 'wb') as f:
        f.write(response.content)
    
    print("Файл скачан. Анализируем формат...")
    
    # Пробуем разные форматы
    try:
        # Попробуем TSV (табуляция) - самый вероятный вариант
        df = pd.read_csv('data/rureviews/raw_reviews.csv', sep='\t', encoding='utf-8')
        print("Формат: TSV (табуляция)")
    except:
        try:
            # Попробуем CSV с ;
            df = pd.read_csv('data/rureviews/raw_reviews.csv', sep=';', encoding='utf-8')
            print("Формат: CSV с ;")
        except:
            try:
                # Попробуем обычный CSV
                df = pd.read_csv('data/rureviews/raw_reviews.csv', encoding='utf-8')
                print("✅ Формат: Обычный CSV")
            except Exception as e:
                print(f"Не удалось определить формат: {e}")
                return
    
    # Переименовываем колонки если нужно
    if len(df.columns) >= 2:
        df.columns = ['text', 'label'][:len(df.columns)]
        print(f"Колонки переименованы: {df.columns.tolist()}")
    
    # Сохраняем в правильном формате
    df.to_csv('data/rureviews/reviews.csv', index=False, encoding='utf-8')
    
    print(f"RuReviews обработан и сохранен!")
    print(f"Размер: {len(df)} примеров")
    print(f"Метки: {df['label'].value_counts().sort_index().to_dict()}")
    
    # Покажем примеры
    print(f"Примеры данных:")
    for i in range(min(3, len(df))):
        text_preview = df.iloc[i]['text'][:100] + '...' if len(str(df.iloc[i]['text'])) > 100 else df.iloc[i]['text']
        print(f"   {i+1}. Текст: {text_preview}")
        print(f"      Метка: {df.iloc[i]['label']}")

if __name__ == "__main__":
    download_rureviews()