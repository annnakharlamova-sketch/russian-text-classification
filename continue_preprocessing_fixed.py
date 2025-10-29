import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
import yaml
import pandas as pd
import os

print("=== ЗАВЕРШЕНИЕ ПРЕДОБРАБОТКИ ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Обрабатываем ТОЛЬКО P3 для rusentiment
corpus_name = 'rusentiment'
pipeline = 'P3'

print(f"\nЗавершение обработки: {corpus_name} - {pipeline}")

# Загружаем исходные данные
corpus_config = config['data']['corpora'][corpus_name]
df = pd.read_csv(corpus_config['path'] + '/train.csv')

print(f"Исходные данные: {len(df)} примеров")

# Создаем конфиг с нужным пайплайном
pipeline_config = config.copy()
pipeline_config['preprocessing']['current_pipeline'] = pipeline

# Создаем препроцессор
preprocessor = DataPreprocessor(pipeline_config)

# Обрабатываем корпус (возвращает словарь)
print("Начало обработки P3 (лемматизация)...")
processed_dict = preprocessor.process_corpus(corpus_name, df)

# Преобразуем словарь в DataFrame для сохранения
processed_data = pd.DataFrame({
    'text': processed_dict['processed_text'],
    'label': processed_dict['labels'] if 'labels' in processed_dict else df['sentiment']
})

# Сохраняем
output_dir = f"processed_data/{corpus_name}"
os.makedirs(output_dir, exist_ok=True)

processed_data.to_csv(f"{output_dir}/{pipeline}.csv", index=False)
print(f" Сохранено: {output_dir}/{pipeline}.csv ({len(processed_data)} примеров)")

print("\n ПРЕДОБРАБОТКА RUSENTIMENT ЗАВЕРШЕНА!")

# Проверим все обработанные файлы
print("\n=== ПРОВЕРКА РЕЗУЛЬТАТОВ ===")
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"Файлы в {output_dir}:")
    for file in files:
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} - {size} байт")