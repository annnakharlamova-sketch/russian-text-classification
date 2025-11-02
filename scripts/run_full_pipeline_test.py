
"""
Полный тест пайплайна на toy-данных
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config
from data_preprocessing import DataPreprocessor
from models import ClassicalModel
from evaluation import Evaluator
import pandas as pd

def run_full_pipeline_test():
    """Запуск полного пайплайна на toy-данных"""
    print(" ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА НА TOY-ДАННЫХ")
    
    config = load_config('configs/experiment_config.yaml')
    
    # 1. Предобработка
    print("1. Предобработка данных...")
    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.process_all_corpora()
    
    # 2. Обучение моделей на RuReviews (самый простой датасет)
    print("2. Обучение моделей...")
    rureviews_data = processed_data['rureviews']['P0']
    
    # Берем подвыборку для скорости
    sample_data = rureviews_data.sample(100, random_state=42)
    
    X = sample_data['processed_text'].tolist()
    y = sample_data['label'].tolist()
    
    # Разделяем на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Обучаем классические модели
    classical_model = ClassicalModel(config)
    classical_model.train_all_models(X_train, y_train)
    
    # 3. Оценка
    print("3. Оценка моделей...")
    evaluator = Evaluator(config)
    
    results = {}
    for model_name in ['bow_logreg', 'tfidf_svm']:
        metrics = classical_model.evaluate_model(model_name, X_test, y_test)
        if metrics:
            results[model_name] = metrics
            print(f"   {model_name}: accuracy={metrics['accuracy']:.3f}")
    
    print(" ПОЛНЫЙ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    return True

if __name__ == "__main__":
    success = run_full_pipeline_test()
    sys.exit(0 if success else 1)