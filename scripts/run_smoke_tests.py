
"""
Smoke-тесты для проверки работоспособности пайплайна
"""

import sys
import os
import pandas as pd
from sklearn.metrics import f1_score

# Добавляем путь к src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Импорты
from src.utils import load_config, ensure_dir
from src.data_preprocessing import DataPreprocessor  
from src.models import ClassicalModel
from src.evaluation import Evaluator

def test_data_loading():
    """Тест загрузки данных"""
    print(" Тест загрузки данных...")
    
    try:
        config = load_config('configs/experiment_config.yaml')
        preprocessor = DataPreprocessor(config)
        
        datasets = ['rureviews', 'rusentiment', 'taiga_social']
        
        for dataset_name in datasets:
            if dataset_name == 'taiga_social':
                # Для smoke-тестов используем ограниченный режим
                data = preprocessor.load_taiga(
                    config['data']['corpora']['taiga_social']['path'],
                    max_sentences=1000,  # Только 1000 предложений
                    skip_large_files=True  # Пропускаем большие файлы
                )
            elif dataset_name == 'rusentiment':
                data = preprocessor.load_rusentiment(config['data']['corpora']['rusentiment']['path'])
            elif dataset_name == 'rureviews':
                data_path = config['data']['corpora']['rureviews']['path']
                data_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
                print(f"    Загрузка RuReviews из: {data_dir}")
                data = preprocessor.load_rureviews(data_dir)
            
            if data is not None and len(data) > 0:
                print(f"    {dataset_name}: {len(data)} строк, {data.columns.tolist()}")
            else:
                print(f"    {dataset_name}: данные не загружены")
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
            lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)  # Прямой вызов f1_score
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