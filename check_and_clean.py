#!/usr/bin/env python3
"""
Скрипт для проверки и очистки данных
"""

import pandas as pd
import os

def check_rureviews():
    """Проверка RuReviews данных"""
    print("Проверка RuReviews...")
    
    try:
        # Пробуем загрузить обработанный файл
        df = pd.read_csv('data/rureviews/reviews.csv')
        
        print("Файл 'reviews.csv' загружается корректно!")
        print(f"Размер: {len(df)} строк")
        print(f"Колонки: {df.columns.tolist()}")
        
        # Проверяем метки
        print(f"Распределение меток:")
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"   - Метка {label}: {count} примеров ({count/len(df)*100:.1f}%)")
        
        # Покажем примеры
        print(f"Примеры данных (первые 3):")
        for i in range(min(3, len(df))):
            text = str(df.iloc[i]['text'])
            text_preview = text[:80] + '...' if len(text) > 80 else text
            print(f"   {i+1}. Текст: {text_preview}")
            print(f"      Метка: {df.iloc[i]['label']}")
            
        return True
        
    except Exception as e:
        print(f"Ошибка загрузки 'reviews.csv': {e}")
        return False

def check_rusentiment():
    """Проверка RuSentiment данных"""
    print("\nПроверка RuSentiment...")
    
    try:
        df = pd.read_csv('data/rusentiment/train.csv')
        print("Файл 'train.csv' загружается корректно!")
        print(f"Размер: {len(df)} строк")
        print(f"Колонки: {df.columns.tolist()}")
        
        # Покажем примеры
        print(f"Первые 2 строки:")
        print(df.head(2))
        
        return True
        
    except Exception as e:
        print(f"Ошибка загрузки RuSentiment: {e}")
        return False

def clean_duplicate_files():
    """Очистка дубликатов файлов"""
    print("\n Очистка дубликатов...")
    
    files_to_keep = ['reviews.csv']  # Оставляем только обработанный файл
    files_to_delete = []
    
    for file in os.listdir('data/rureviews'):
        if file not in files_to_keep and file.endswith('.csv'):
            files_to_delete.append(file)
    
    for file in files_to_delete:
        try:
            file_path = os.path.join('data/rureviews', file)
            file_size = os.path.getsize(file_path) / 1024 / 1024  # Размер в МБ
            os.remove(file_path)
            print(f"Удален: {file} ({file_size:.1f} МБ)")
        except Exception as e:
            print(f" Ошибка удаления {file}: {e}")
    
    return len(files_to_delete)

def main():
    """Главная функция"""
    print("Проверка и очистка данных")
    print("=" * 50)
    
    # Проверяем данные
    rureviews_ok = check_rureviews()
    rusentiment_ok = check_rusentiment()
    
    # Очищаем дубликаты
    if rureviews_ok:
        deleted_count = clean_duplicate_files()
        print(f"\n Удалено файлов: {deleted_count}")
    
    # Итог
    print("\n" + "=" * 50)
    print("ИТОГ ПРОВЕРКИ:")
    print(f"   RuReviews: {'ГОТОВ' if rureviews_ok else 'ОШИБКА'}")
    print(f"   RuSentiment: {'ГОТОВ' if rusentiment_ok else 'ОШИБКА'}")
    
    if rureviews_ok and rusentiment_ok:
        print("Все данные готовы для экспериментов!")
    else:
        print("Есть проблемы с данными")

if __name__ == "__main__":
    main()