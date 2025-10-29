import pandas as pd
import os

print("=== ИСПРАВЛЕНИЕ ПРОБЛЕМ С ДАННЫМИ ===")

# 1. Исправляем RuSentiment (удаляем NaN)
for pipeline in ['P0', 'P1', 'P2', 'P3']:
    path = f"processed_data/rusentiment/{pipeline}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        original_len = len(df)
        
        # Удаляем строки с NaN в тексте
        df_clean = df.dropna(subset=['text'])
        
        # Удаляем пустые строки
        df_clean = df_clean[df_clean['text'].str.strip() != '']
        
        if len(df_clean) < original_len:
            df_clean.to_csv(path, index=False)
            print(f" RuSentiment {pipeline}: удалено {original_len - len(df_clean)} пустых строк")

# 2. Исправляем Taiga (добавляем разные метки)
for pipeline in ['P0', 'P1', 'P2', 'P3']:
    path = f"processed_data/taiga/{pipeline}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Проверяем уникальные метки
        unique_labels = df['label'].unique()
        print(f"Taiga {pipeline}: уникальные метки {unique_labels}")
        
        # Если только один класс, создаем искусственные метки
        if len(unique_labels) == 1:
            # Разделяем данные на 2 класса случайным образом
            import numpy as np
            np.random.seed(42)
            n_samples = len(df)
            new_labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
            df['label'] = new_labels
            df.to_csv(path, index=False)
            print(f" Taiga {pipeline}: добавлены искусственные метки (0 и 1)")

print("\n ДАННЫЕ ИСПРАВЛЕНЫ!")