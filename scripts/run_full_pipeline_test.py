"""
Полный тест пайплайна на toy-данных
"""

import sys
import os
import pandas as pd
import numpy as np

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Теперь импортируем через src
from src.utils import load_config
from src.data_preprocessing import DataPreprocessor
from src.models import ClassicalModel
from src.evaluation import Evaluator

def create_toy_data_for_pipeline():
    """Создание toy-данных специально для пайплайна"""
    print("Создание toy-данных для пайплайна...")
    
    # Создаем реалистичные данные для всех корпусов
    toy_data = {}
    
    # RuReviews данные
    rureviews_data = pd.DataFrame({
        'text': [
            'Отличный товар! Очень доволен покупкой 123.',
            'Плохое качество, не рекомендую :(',
            'Нормально за свои деньги...',
            'Прекрасный сервис и быстрая доставка!',
            'Ужасное обслуживание, больше не обращусь.',
            'Хороший продукт, соответствует описанию.',
            'Разочарован, ожидал большего.',
            'Быстро доставили, спасибо!',
            'Некачественный товар, вернул.',
            'Отлично! Советую всем.',
            'Качество на высоте, рекомендую к покупке!',
            'Не понравилось, не стоит денег.',
            'Быстро, качественно, вежливо!',
            'Долго ждал доставку, недоволен.',
            'Отличное соотношение цена/качество.'
        ],
        'label': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    })
    toy_data['rureviews'] = rureviews_data
    
    # RuSentiment данные
    rusentiment_data = pd.DataFrame({
        'text': [
            'Это просто прекрасно! Очень рад.',
            'Ненавижу этот продукт. Ужасное качество.',
            'Нормальный сервис, ничего особенного.',
            'Восхитительная работа! Очень доволен.',
            'Ужасное качество. Не покупайте.',
            'Хороший продукт за свои деньги.',
            'Разочаровался в покупке.',
            'Отличное обслуживание! Рекомендую.',
            'Плохое качество, не стоит денег.',
            'Прекрасный товар! Очень доволен.'
        ],
        'sentiment': [2, 0, 1, 2, 0, 1, 0, 2, 0, 2]
    })
    toy_data['rusentiment'] = rusentiment_data
    
    print(f"Создано toy-данных: {len(rureviews_data)} RuReviews, {len(rusentiment_data)} RuSentiment")
    return toy_data

def run_full_pipeline_test():
    """Запуск полного пайплайна на toy-данных"""
    print(" ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА НА TOY-ДАННЫХ")
    print("=" * 60)
    
    try:
        config = load_config('configs/experiment_config.yaml')
        print(" Конфигурация загружена")
    except Exception as e:
        print(f" Ошибка загрузки конфигурации: {e}")
        return False
    
    try:
        # 1. Создание и предобработка toy-данных
        print("\n1. Предобработка данных...")
        preprocessor = DataPreprocessor(config)
        toy_data = create_toy_data_for_pipeline()
        
        # Обрабатываем данные через пайплайны
        processed_data = {}
        for corpus_name, data in toy_data.items():
            print(f"   Обработка {corpus_name}...")
            
            # Переименование колонок если нужно
            if corpus_name == 'rusentiment' and 'sentiment' in data.columns:
                data = data.rename(columns={'sentiment': 'label'})
            
            # Применяем пайплайны предобработки
            corpus_results = {}
            for pipeline_name in ['P0', 'P1']:  # Только базовые пайплайны для теста
                processed = data.copy()
                processed['processed_text'] = data['text'].apply(
                    lambda x: preprocessor.apply_pipeline(x, pipeline_name)
                )
                processed = processed[processed['processed_text'].str.len() > 0]
                corpus_results[pipeline_name] = processed
            
            processed_data[corpus_name] = corpus_results
            print(f"   ✓ {corpus_name}: {len(data)} примеров")
        
        print("✓ Данные предобработаны")
        
    except Exception as e:
        print(f"✗ Ошибка предобработки: {e}")
        return False
    
    try:
        # 2. Обучение моделей на RuReviews данных
        print("\n2. Обучение моделей...")
        
        if 'rureviews' not in processed_data:
            print("✗ Нет данных RuReviews для обучения")
            return False
            
        rureviews_data = processed_data['rureviews']['P0']  # Используем P0 пайплайн
        
        X = rureviews_data['processed_text'].tolist()
        y = rureviews_data['label'].tolist()
        
        # Разделяем на train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"   Данные для обучения: {len(X_train)} train, {len(X_test)} test")
        
        # Обучаем классические модели
        classical_model = ClassicalModel(config)
        success = classical_model.train_all_models(X_train, y_train)
        
        if not success:
            print("✗ Ошибка обучения моделей")
            return False
            
        print("✓ Модели обучены")
        
    except Exception as e:
        print(f"✗ Ошибка обучения моделей: {e}")
        return False
    
    try:
        # 3. Оценка моделей
        print("\n3. Оценка моделей...")
        evaluator = Evaluator(config)
        
        results = {}
        for model_name in ['bow_logreg', 'tfidf_svm']:
            print(f"   Тестирование {model_name}...")
            metrics = classical_model.evaluate_model(model_name, X_test, y_test)
            if metrics:
                results[model_name] = metrics
                print(f"     ✓ {model_name}: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}")
            else:
                print(f"     ✗ {model_name}: ошибка оценки")
        
        print("✓ Модели оценены")
        
    except Exception as e:
        print(f"✗ Ошибка оценки моделей: {e}")
        return False
    
    try:
        # 4. Сохранение результатов
        print("\n4. Сохранение результатов...")
        
        # Сохраняем модели
        classical_model.save_models('trained_models/test_pipeline')
        print("   ✓ Модели сохранены")
        
        # Сохраняем метрики
        if results:
            results_df = pd.DataFrame(results).T
            os.makedirs('results/test_pipeline', exist_ok=True)
            results_df.to_csv('results/test_pipeline/metrics.csv')
            print("   ✓ Метрики сохранены")
        
        print("✓ Результаты сохранены")
        
    except Exception as e:
        print(f"✗ Ошибка сохранения: {e}")
        return False
    
    print("\n" + "=" * 60)
    print(" ПОЛНЫЙ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print(f"   Обучено моделей: {len(results)}")
    print(f"   Протестировано примеров: {len(X_test)}")
    
    return True

if __name__ == "__main__":
    success = run_full_pipeline_test()
    sys.exit(0 if success else 1)