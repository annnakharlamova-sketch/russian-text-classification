import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
import yaml
import pandas as pd
import os

print("=== ФИНАЛЬНАЯ ПРЕДОБРАБОТКА ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Загружаем ВСЕ данные Rusentiment
corpus_name = 'rusentiment'
df = pd.read_csv('data/rusentiment/train.csv')

print(f"Обработка: {corpus_name} - {len(df)} примеров")

# Создаем препроцессор
preprocessor = DataPreprocessor(config)

print("Начало обработки всех пайплайнов...")
# Обрабатываем ВСЕ пайплайны за один раз
results = preprocessor.process_corpus(corpus_name, df)

print("Сохранение результатов...")
# Сохраняем каждый пайплайн
output_dir = f"processed_data/{corpus_name}"
os.makedirs(output_dir, exist_ok=True)

for pipeline, processed_df in results.items():
    # Сохраняем только текст и метку
    save_df = pd.DataFrame({
        'text': processed_df['text'],
        'label': df['sentiment']  # Берем оригинальные метки
    })
    
    save_df.to_csv(f"{output_dir}/{pipeline}.csv", index=False)
    print(f"Сохранено: {output_dir}/{pipeline}.csv ({len(save_df)} примеров)")

print("\nВСЯ ПРЕДОБРАБОТКА RUSENTIMENT ЗАВЕРШЕНА!")

# Проверим результаты
print("\n=== ИТОГИ ===")
files = os.listdir(output_dir)
total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
print(f"Файлов: {len(files)}")
print(f"Общий размер: {total_size / (1024*1024):.1f} MB")
for file in files:
    size = os.path.getsize(os.path.join(output_dir, file))
    print(f"  {file}: {size / (1024*1024):.1f} MB")