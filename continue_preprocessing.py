import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
import yaml
import pandas as pd
import os

print("=== ПРОДОЛЖЕНИЕ ПРЕДОБРАБОТКИ ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))
preprocessor = DataPreprocessor(config)

# Обрабатываем ТОЛЬКО оставшиеся корпуса и пайплайны
corpora_to_process = ['rusentiment']  # Только то, что не завершено
pipelines_to_process = ['P2', 'P3']   # Только оставшиеся пайплайны

for corpus_name in corpora_to_process:
    print(f"\nОбработка корпуса: {corpus_name}")
    
    # Загружаем исходные данные
    corpus_config = config['data']['corpora'][corpus_name]
    df = pd.read_csv(corpus_config['path'] + '/train.csv')
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    print(f"Исходные данные: {len(texts)} примеров")
    
    # Обрабатываем только нужные пайплайны
    for pipeline in pipelines_to_process:
        print(f"  Пайплайн: {pipeline}")
        processed_texts = preprocessor.process_corpus(texts, pipeline=pipeline)
        
        # Сохраняем
        output_dir = f"processed_data/{corpus_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        result_df = pd.DataFrame({
            'text': processed_texts,
            'label': labels
        })
        result_df.to_csv(f"{output_dir}/{pipeline}.csv", index=False)
        print(f"  Сохранено: {output_dir}/{pipeline}.csv ({len(result_df)} примеров)")

print("\nПредобработка завершена!")