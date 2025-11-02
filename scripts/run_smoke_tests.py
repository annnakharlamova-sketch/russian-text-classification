
"""
Smoke-тесты для проверки работоспособности пайплайна
"""

import sys
import os
import pandas as pd

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, ensure_dir
from data_preprocessing import DataPreprocessor
from models import ClassicalModel
from evaluation import Evaluator

def test_data_loading():
    """Тест загрузки данных"""
    print(" Тест загрузки данных...")
    
    try:
        # Проверяем существование toy-датасетов
        datasets = {
            'rureviews': 'data/rureviews/reviews.csv',
            'rusentiment': 'data/rusentiment/train.csv', 
            'taiga': 'data/taiga_extracted/social/toy_social.csv'
        }
        
        for name, path in datasets.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"    {name}: {len(df)} строк, {df.columns.tolist()}")
            else:
                print(f"    {name}: файл не найден")
                return False
                
        return True
        
    except Exception as e:
        print(f"    Ошибка: {e}")
        return False

def test_preprocessing():
    """Тест предобработки"""
    print(" Тест предобработки...")
    
    try:
        config = load_config('configs/experiment_config.yaml')
        preprocessor = DataPreprocessor(config)
        
        # Тестируем на маленьком наборе данных
        test_texts = [
            "Отличный товар! Очень доволен 123.",
            "Плохое качество :( Не рекомендую...",
            "Нормально, но есть недочеты 456."
        ]
        
        # Тестируем разные пайплайны
        pipelines = ['P0', 'P1', 'P2', 'P3']
        
        for pipeline in pipelines:
            processed = []
            for text in test_texts:
                result = preprocessor.apply_pipeline(text, pipeline)
                processed.append(result)
            
            print(f"    {pipeline}: {processed[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"    Ошибка: {e}")
        return False

def test_classical_models():
    """Тест классических моделей"""
    print(" Тест классических моделей...")
   
    try:
        config = load_config('configs/experiment_config.yaml')
        
        # Тестовые данные
        X_train = [
            "отличный товар качество хорошее",
            "плохой продукт не советую",
            "нормально соответствует описанию", 
            "хорошо быстрая доставка",
            "ужасно долго ждал"
        ]
        y_train = [1, 0, 1, 1, 0]
        
        X_test = ["хороший продукт", "плохое качество"]
        y_test = [1, 0]
        
        # Тестируем обучение
        model = ClassicalModel(config)
        success = model.train_all_models(X_train, y_train)
        
        if not success:
            print("    Ошибка обучения моделей")
            return False
        
        # Тестируем оценку
        for model_name in ['bow_logreg', 'tfidf_svm']:
            metrics = model.evaluate_model(model_name, X_test, y_test)
            if metrics:
                print(f"    {model_name}: accuracy={metrics['accuracy']:.3f}")
            else:
                print(f"    {model_name}: ошибка оценки")
                return False
                
        return True
        
    except Exception as e:
        print(f"    Ошибка: {e}")
        return False

def test_evaluation():
    """Тест системы оценки"""
    print(" Тест системы оценки...")
    
    try:
        config = load_config('configs/experiment_config.yaml')
        evaluator = Evaluator(config)
        
        # Тестовые данные
        y_true = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
        
        # Тест базовых метрик
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        print(f"    Метрики: accuracy={metrics['accuracy']:.3f}")
        
        # Тест доверительных интервалов
        mean_f1, ci = evaluator.bootstrap_confidence_interval(
            y_true, y_pred, 
            lambda yt, yp: evaluator.calculate_metrics(yt, yp)['f1_macro']
        )
        print(f"    Доверительный интервал: {ci[0]:.3f}-{ci[1]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"    Ошибка: {e}")
        return False

def main():
    """Главная функция smoke-тестов"""
    print(" ЗАПУСК SMOKE-ТЕСТОВ")
    print("=" * 50)
    
    tests = [
        ("Загрузка данных", test_data_loading),
        ("Предобработка", test_preprocessing), 
        ("Классические модели", test_classical_models),
        ("Система оценки", test_evaluation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"    {test_name}: КРИТИЧЕСКАЯ ОШИБКА - {e}")
            results.append((test_name, False))
    
    # Вывод результатов
    print("\n" + "=" * 50)
    print(" РЕЗУЛЬТАТЫ SMOKE-ТЕСТОВ:")
    
    all_passed = True
    for test_name, success in results:
        status = " ПРОЙДЕН" if success else " ПРОВАЛЕН"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print(" ВСЕ SMOKE-ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        sys.exit(0)
    else:
        print(" НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ!")
        sys.exit(1)

if __name__ == "__main__":
    main()