
"""
Генерация данных для кривых обучения
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

def generate_learning_curve_data():
    """Генерация данных для кривых обучения"""
    print(" Генерация данных для кривых обучения...")
    
    # Демо-данные (в реальном использовании загружайте настоящие данные)
    # Здесь просто создаем пример структуры
    learning_data = []
    
    datasets = ['rusentiment', 'rureviews', 'taiga_social']
    models = ['bow_logreg', 'tfidf_svm']
    
    for dataset in datasets:
        for model in models:
            # Демо данные кривых обучения
            train_sizes = [1000, 5000, 10000, 20000, 50000]
            
            for size in train_sizes:
                # Демо значения scores
                train_score = 0.6 + 0.3 * (1 - np.exp(-size / 20000))
                test_score = 0.55 + 0.25 * (1 - np.exp(-size / 20000))
                
                learning_data.append({
                    'dataset': dataset,
                    'model': model,
                    'train_size': size,
                    'train_score': train_score + np.random.normal(0, 0.02),
                    'test_score': test_score + np.random.normal(0, 0.02)
                })
    
    # Сохранение
    lc_df = pd.DataFrame(learning_data)
    output_path = Path('results/learning_curves.csv')
    lc_df.to_csv(output_path, index=False)
    print(f" Данные кривых обучения сохранены: {output_path}")
    
    return lc_df

if __name__ == "__main__":
    generate_learning_curve_data()