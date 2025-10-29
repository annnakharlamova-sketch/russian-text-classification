import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
import yaml
import pandas as pd
import os

print("=== ОТЛАДКА СТРУКТУРЫ ДАННЫХ ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Берем только 10 примеров для теста
df = pd.read_csv('data/rusentiment/train.csv').head(10)

print(f"Тестовые данные: {len(df)} примеров")

# Создаем препроцессор
pipeline_config = config.copy()
pipeline_config['preprocessing']['current_pipeline'] = 'P3'
preprocessor = DataPreprocessor(pipeline_config)

# Обрабатываем маленькую порцию и смотрим структуру
print("Обработка 10 примеров...")
result = preprocessor.process_corpus('rusentiment', df)

print("=== СТРУКТУРА РЕЗУЛЬТАТА ===")
print(f"Тип: {type(result)}")
print(f"Ключи: {list(result.keys()) if isinstance(result, dict) else 'Не словарь'}")

if isinstance(result, dict):
    for key, value in result.items():
        print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
        if key in ['processed_text', 'text'] and len(value) > 0:
            print(f"    Пример: {value[0][:100]}...")

print("=== СТРУКТУРА ИСХОДНОГО DataFrame ===")
print(f"Колонки: {list(df.columns)}")
print(f"Первая строка: {dict(df.iloc[0])}")