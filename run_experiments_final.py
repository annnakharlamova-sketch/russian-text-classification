#!/usr/bin/env python3
"""
Полный пайплайн экспериментов классификации текстов
"""

import argparse
import sys
import os
import pandas as pd

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config, print_config_summary, ensure_dir, setup_reproducibility
from src.data_preprocessing import DataPreprocessor
from src.models import ClassicalModel, RealNeuralModel
from src.evaluation import Evaluator
from src.analysis import PreprocessingAnalyzer


def load_processed_data(corpus_name, pipeline_name):
    """Загрузка обработанных данных"""
    file_path = f"processed_data/{corpus_name}/{pipeline_name}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f" Загружены данные: {file_path} ({len(df)} примеров)")
        return df
    else:
        print(f" Файл не найден: {file_path}")
        return None


def train_classical_models(config, X_train, y_train, model_name_suffix=""):
    """Обучение классических моделей"""
    print(f" Обучение классических моделей{model_name_suffix}...")
    
    classical_model = ClassicalModel(config)
    success = classical_model.train_all_models(X_train, y_train)
    
    if success:
        output_dir = f"trained_models/classical{model_name_suffix}"
        classical_model.save_models(output_dir)
        print(f" Классические модели сохранены в: {output_dir}")
    
    return classical_model if success else None


def train_neural_models(config, X_train, y_train, model_name_suffix=""):
    """Обучение нейросетевых моделей"""
    print(f" Обучение нейросетевых моделей{model_name_suffix}...")
    
    try:
        neural_model = RealNeuralModel(config)
        success = neural_model.train_all_models(X_train, y_train)
        
        if success:
            print(f" Нейросетевые модели обучены{model_name_suffix}")
        
        return neural_model if success else None
    except Exception as e:
        print(f" Ошибка нейросетевых моделей: {e}")
        return None


def run_complete_evaluation(config, classical_model=None, neural_model=None, X_test=None, y_test=None, 
                          eval_name="", X_train=None, y_train=None, dataset_name=None, 
                          model_name=None, preprocess_name=None):
    """
    Полная оценка моделей с поддержкой разных сценариев
    """
    print(f" Полная оценка моделей{eval_name}...")
    
    evaluator = Evaluator(config)
    predictions_dict = {}
    results = {}
    
    # Сценарий 1: Оценка классических моделей
    if classical_model and X_test is not None and y_test is not None:
        for model_name in ['bow_logreg', 'tfidf_svm']:
            if model_name in classical_model.models:
                print(f"   Оценка {model_name}...")
                vectorizer = classical_model.vectorizers[model_name]
                model = classical_model.models[model_name]
                
                metrics, y_pred = evaluator.evaluate_classical_model(
                    model, vectorizer, X_test, y_test, model_name
                )
                predictions_dict[model_name] = y_pred
                results[model_name] = metrics
    
    # Сценарий 2: Кросс-валидация для конкретной модели
    elif X_train is not None and y_train is not None and model_name is not None:
        print(f"   Кросс-валидация {model_name}...")
        
        # Здесь должна быть передана модель как параметр
        # Для демонстрации создадим временную модель
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        
        # 5-кратная стратифицированная CV
        cv_results, avg_metrics = evaluator.cross_validate_model(
            model, X_train, y_train, dataset_name, model_name, preprocess_name
        )
        
        # Оценка на тестовом наборе
        if X_test is not None and y_test is not None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_result = evaluator.evaluate_with_confidence_intervals(
                y_test, y_pred, dataset_name, model_name, preprocess_name
            )
            results['test'] = test_result
        
        results['cv'] = avg_metrics
    
    # Сравнение моделей (если есть несколько моделей)
    if len(predictions_dict) >= 2:
        print(f"\n Сравнение моделей{eval_name}:")
        comparison_results = evaluator.compare_models(y_test, predictions_dict)
        results_table = evaluator.create_results_table(comparison_results)
        results['comparison'] = comparison_results
        
        return comparison_results, results_table
    
    return results


def analyze_preprocessing_impact(config, processed_data):
    """Анализ влияния предобработки"""
    print(" Анализ влияния предобработки...")
    
    analyzer = PreprocessingAnalyzer(config)
    
    # Анализ сокращения словаря
    analysis_results = analyzer.analyze_vocabulary_reduction(processed_data)
    
    # Создание таблиц и графиков
    analyzer.create_preprocessing_table(analysis_results)
    analyzer.plot_preprocessing_impact(analysis_results)
    
    print(" Анализ предобработки завершен!")
    return analysis_results


def main():
    print(" ЗАПУСК ФИНАЛЬНОГО ЭКСПЕРИМЕНТА")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='Полный пайплайн экспериментов')
    parser.add_argument('--all', action='store_true', help='Запустить все эксперименты')
    parser.add_argument('--preprocess', action='store_true', help='Только предобработка')
    parser.add_argument('--classical', action='store_true', help='Только классические модели')
    parser.add_argument('--neural', action='store_true', help='Только нейросетевые модели')
    parser.add_argument('--evaluate', action='store_true', help='Только оценка')
    parser.add_argument('--analyze', action='store_true', help='Только анализ')
    parser.add_argument('--config', default='configs/experiment_config.yaml', help='Путь к конфигурации')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (по умолчанию: 42)')
    
    args = parser.parse_args()
    
    print(f"Установка random seed: {args.seed}")
    setup_reproducibility(args.seed)

    # Если не указаны аргументы, показываем помощь
    if not any([args.all, args.preprocess, args.classical, args.neural, args.evaluate, args.analyze]):
        print("  Использование: python run_experiments_final.py --all")
        parser.print_help()
        return
    
    # Загрузка конфигурации
    config = load_config(args.config)
    print_config_summary(config)
    
    # Создание папок для результатов
    ensure_dir('results/figures')
    ensure_dir('results/tables')
    ensure_dir('trained_models')
    
    # Переменные для хранения результатов
    processed_data = None
    classical_model = None
    neural_model = None
    
    # 1. ПРЕДОБРАБОТКА ДАННЫХ
    if args.all or args.preprocess:
        print("\n" + "="*60)
        print(" ЭТАП 1: ПРЕДОБРАБОТКА ДАННЫХ")
        print("="*60)
        
        preprocessor = DataPreprocessor(config)
        processed_data = preprocessor.process_all_corpora()
        
        # Анализ влияния предобработки
        if processed_data:
            analyze_preprocessing_impact(config, processed_data)
    
    # 2. ОБУЧЕНИЕ МОДЕЛЕЙ НА ТЕСТОВЫХ ДАННЫХ
    if args.all or args.classical or args.neural:
        print("\n" + "="*60)
        print(" ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ (ТЕСТОВЫЕ ДАННЫЕ)")
        print("="*60)
        
        # Тестовые данные для демонстрации
        X_train = [
            "отличный товар очень доволен качеством доставка быстрая",
            "ужасное качество не рекомендую к покупке товар бракованный",
            "нормальный продукт за свои деньги соответствует описанию",
            "прекрасное обслуживание спасибо большое вежливый персонал",
            "кошмарный сервис больше не обращусь никогда плохая связь",
            "хороший продукт советую всем друзьям отличное качество",
            "некачественный товар деньги на ветер не работает",
            "удовлетворительно но есть недочеты можно лучше",
            "великолепно быстро доставили упаковка целая",
            "ужасно долгая доставка поврежденная упаковка"
        ]
        y_train = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        
        X_test = [
            "хороший продукт советую всем качество на уровне",
            "плохое качество разочарован покупкой не советую",
            "отлично быстрая доставка хороший сервис",
            "недоволен обслуживанием медленно реагируют"
        ]
        y_test = [1, 0, 1, 0]
        
        # Классические модели
        if args.all or args.classical:
            classical_model = train_classical_models(config, X_train, y_train)
        
        # Нейросетевые модели
        if args.all or args.neural:
            neural_model = train_neural_models(config, X_train, y_train)
        
        # Оценка на тестовых данных
        # Оценка на тестовых данных
        if args.all or args.evaluate:
            results = run_complete_evaluation(
                config, 
                classical_model=classical_model, 
                neural_model=neural_model, 
                X_test=X_test, 
                y_test=y_test,
                eval_name=" (тестовые данные)"
            )
            print("Результаты оценки:", results)
    
    # 3. ЭКСПЕРИМЕНТЫ С РЕАЛЬНЫМИ ДАННЫМИ (если они обработаны)
    if processed_data and (args.all or args.evaluate):
        print("\n" + "="*60)
        print(" ЭТАП 3: ЭКСПЕРИМЕНТЫ С РЕАЛЬНЫМИ ДАННЫМИ")
        print("="*60)
        
        # Здесь можно добавить эксперименты с реальными данными
        # Например: сравнение разных пайплайнов предобработки
        
        print("  Реальные данные готовы для экспериментов!")
        print("   Используйте load_processed_data() для загрузки конкретных пайплайнов")
    
    print("\n" + "="*60)
    print(" ВСЕ ЭТАПЫ ЭКСПЕРИМЕНТА ЗАВЕРШЕНЫ!")
    print("="*60)
    print("\n РЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
    print("   - results/figures/    (графики)")
    print("   - results/tables/     (таблицы)")
    print("   - trained_models/     (обученные модели)")
    print("   - processed_data/     (обработанные данные)")


if __name__ == "__main__":
    main()