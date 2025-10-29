import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
import yaml
import pandas as pd
import os

print("=== ОБРАБОТКА TAIGA (ОГРАНИЧЕННЫЙ ОБЪЕМ) ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Загружаем данные Taiga с ограничением
print("Загрузка Taiga corpus...")
taiga_path = 'data/taiga_extracted'
taiga_data = []

# Ограничим общее количество примеров
MAX_EXAMPLES = 50000  # Достаточно для исследования

# Читаем текстовые файлы с ограничением
text_files = []
for root, dirs, files in os.walk(taiga_path):
    for file in files:
        if file.endswith('.txt'):
            text_files.append(os.path.join(root, file))

print(f"Найдено текстовых файлов: {len(text_files)}")

# Читаем содержимое файлов с ограничением
total_texts = 0
for file_path in text_files:
    if total_texts >= MAX_EXAMPLES:
        break
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Разбиваем на отдельные тексты
        texts = [t.strip() for t in content.split('\n') if len(t.strip()) > 30]
        
        # Ограничиваем количество из каждого файла
        texts_to_take = min(len(texts), (MAX_EXAMPLES - total_texts) // len(text_files) + 1)
        
        for text in texts[:texts_to_take]:
            taiga_data.append({
                'text': text,
                'label': 0  # временная метка
            })
        
        total_texts += texts_to_take
        print(f"Загружено {texts_to_take} текстов из {os.path.basename(file_path)} (всего: {total_texts})")
        
    except Exception as e:
        print(f"Ошибка чтения {file_path}: {e}")

print(f"Итого загружено: {len(taiga_data)} примеров")

# Создаем DataFrame для обработки
df = pd.DataFrame(taiga_data)

print(f"DataFrame создан: {len(df)} строк, {len(df.columns)} колонок")

# Создаем препроцессор
preprocessor = DataPreprocessor(config)

print("Начало обработки всех пайплайнов...")
# Обрабатываем ВСЕ пайплайны за один раз
results = preprocessor.process_corpus('taiga', df)

print("Сохранение результатов...")
# Сохраняем каждый пайплайн
output_dir = f"processed_data/taiga"
os.makedirs(output_dir, exist_ok=True)

for pipeline, processed_df in results.items():
    # Сохраняем только текст и метку
    save_df = pd.DataFrame({
        'text': processed_df['text'],
        'label': df['label']  # Берем оригинальные метки
    })
    
    save_df.to_csv(f"{output_dir}/{pipeline}.csv", index=False)
    print(f" Сохранено: {output_dir}/{pipeline}.csv ({len(save_df)} примеров)")

print("\n ОБРАБОТКА TAIGA ЗАВЕРШЕНА!")

# Проверим результаты
print("\n=== ИТОГИ TAIGA ===")
files = os.listdir(output_dir)
total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
print(f"Файлов: {len(files)}")
print(f"Общий размер: {total_size / (1024*1024):.1f} MB")
for file in files:
    size = os.path.getsize(os.path.join(output_dir, file))
    lines = sum(1 for _ in open(os.path.join(output_dir, file), 'r', encoding='utf-8')) - 1  # минус заголовок
    print(f"  {file}: {size / (1024*1024):.1f} MB, {lines} примеров")