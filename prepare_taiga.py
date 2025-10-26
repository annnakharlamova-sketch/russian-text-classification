#!/usr/bin/env python3
"""
Скрипт для подготовки Taiga Corpus
"""

import os
import tarfile
import pandas as pd
import json
from pathlib import Path

def extract_taiga_corpus():
    """Распаковка Taiga Corpus"""
    print("Распаковка Taiga Corpus...")
    
    taiga_path = r"C:\Users\Admin\Downloads\social.tar.gz"
    extract_to = "data/taiga_extracted"
    
    if not os.path.exists(taiga_path):
        print(f" Файл не найден: {taiga_path}")
        return False
    
    try:
        # Создаем папку для распаковки
        os.makedirs(extract_to, exist_ok=True)
        
        # Распаковываем архив
        with tarfile.open(taiga_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print(f"Архив распакован в: {extract_to}")
        
        # Покажем что внутри
        print("Содержимое архива:")
        for root, dirs, files in os.walk(extract_to):
            for file in files[:10]:  # Покажем первые 10 файлов
                print(f"   - {os.path.join(root, file)}")
            if files:
                print(f"   ... и еще {len(files) - 10} файлов" if len(files) > 10 else "")
            break
        
        return extract_to
        
    except Exception as e:
        print(f"Ошибка распаковки: {e}")
        return False

def analyze_taiga_structure(extract_path):
    """Анализ структуры Taiga Corpus"""
    print("\n Анализ структуры Taiga Corpus...")
    
    # Ищем JSONL файлы (обычный формат для Taiga)
    jsonl_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.jsonl') or file.endswith('.json'):
                jsonl_files.append(os.path.join(root, file))
    
    print(f"Найдено JSONL файлов: {len(jsonl_files)}")
    
    if not jsonl_files:
        print("Не найдены JSONL файлы. Ищем другие форматы...")
        # Ищем другие форматы
        all_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith(('.txt', '.csv', '.tsv')):
                    all_files.append(os.path.join(root, file))
        print(f"Найдено других файлов: {len(all_files)}")
        return all_files
    
    return jsonl_files

def load_taiga_data(file_path):
    """Загрузка данных из Taiga файла"""
    print(f"\nЗагрузка данных из: {os.path.basename(file_path)}")
    
    try:
        if file_path.endswith('.jsonl'):
            # Читаем JSONL файл
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
            
            print(f"Загружено {len(data)} записей")
            
            # Анализируем структуру
            if data:
                print("Структура данных:")
                first_item = data[0]
                for key in list(first_item.keys())[:5]:  # Покажем первые 5 ключей
                    value_preview = str(first_item[key])[:100] + '...' if len(str(first_item[key])) > 100 else str(first_item[key])
                    print(f"   - {key}: {value_preview}")
            
            return data
            
        else:
            print(f"Формат файла не поддерживается: {file_path}")
            return None
            
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

def create_taiga_dataset(data, output_path):
    """Создание датасета для экспериментов"""
    print(f"\nСоздание датасета Taiga...")
    
    try:
        # Преобразуем в DataFrame
        texts = []
        labels = []
        
        for item in data:
            # Извлекаем текст (зависит от структуры данных)
            if 'text' in item:
                texts.append(item['text'])
            elif 'content' in item:
                texts.append(item['content'])
            elif 'sentence' in item:
                texts.append(item['sentence'])
            else:
                # Берем первую строковую колонку
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        texts.append(value)
                        break
            
            # Извлекаем метку (если есть)
            if 'label' in item:
                labels.append(item['label'])
            elif 'category' in item:
                labels.append(item['category'])
            elif 'sentiment' in item:
                labels.append(item['sentiment'])
            else:
                labels.append(0)  # Заглушка
        
        # Создаем DataFrame
        df = pd.DataFrame({
            'text': texts[:10000],  # Берем первые 10000 примеров для тестов
            'label': labels[:10000]
        })
        
        # Сохраняем
        os.makedirs('data/taiga', exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Датасет создан: {len(df)} примеров")
        print(f"Распределение меток: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Ошибка создания датасета: {e}")
        return None

def main():
    """Главная функция"""
    print("Подготовка Taiga Corpus")
    print("=" * 50)
    
    # 1. Распаковываем архив
    extract_path = extract_taiga_corpus()
    if not extract_path:
        return
    
    # 2. Анализируем структуру
    data_files = analyze_taiga_structure(extract_path)
    
    if not data_files:
        print("Не найдены файлы с данными")
        return
    
    # 3. Загружаем первый найденный файл
    taiga_data = load_taiga_data(data_files[0])
    
    if taiga_data:
        # 4. Создаем датасет для экспериментов
        output_path = "data/taiga/taiga_dataset.csv"
        df = create_taiga_dataset(taiga_data, output_path)
        
        if df is not None:
            print(f"\nTaiga Corpus готов для экспериментов!")
            print(f"Файл: {output_path}")
            print(f"Размер: {len(df)} примеров")
    
    print("\n" + "=" * 50)
    print("Подготовка Taiga завершена!")

if __name__ == "__main__":
    main()