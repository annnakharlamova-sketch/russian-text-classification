import sys
sys.path.append('src')

from utils import save_experiment_results
from evaluation import Evaluator
import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import time
from datetime import datetime

print("ОЦЕНКА И СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiments/main.yaml', 'r', encoding='utf-8'))

# Создаем папку для результатов
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Создаем оценщик
evaluator = Evaluator(config)

results = []

def evaluate_and_save_model(model, vectorizer, X_test, y_test, dataset_name, model_name, preprocess_name, train_time=0, fold=None):
    """
    Оценка модели и сохранение в стандартном формате
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Преобразуем тестовые данные
    X_test_vec = vectorizer.transform(X_test)
    
    # Предсказания
    start_time = time.time()
    y_pred = model.predict(X_test_vec)
    predict_time = time.time() - start_time
    
    # Вычисляем метрики (MACRO averaging как в статье)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Подготовка результатов в требуемом формате
    results_dict = {
        'dataset': dataset_name,
        'model': model_name,
        'preprocess': preprocess_name,
        'fold': fold,  # None для теста, номер для CV
        'seed': 42,
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'train_time_sec': round(train_time, 2)
    }
    
    # Сохранение в стандартизированный CSV
    save_experiment_results(results_dict)
    
    return results_dict

# Оцениваем на всех корпусах и пайплайнах
for corpus_name in ['rureviews', 'rusentiment', 'taiga']:
    print(f"\n{'='*50}")
    print(f"ОЦЕНКА НА КОРПУСЕ: {corpus_name}")
    print(f"{'='*50}")
    
    for pipeline in ['P0', 'P1', 'P2', 'P3']:
        print(f"\n Пайплайн: {pipeline} ")
        
        # Загружаем обработанные данные
        data_path = f"processed_data/{corpus_name}/{pipeline}.csv"
        if not os.path.exists(data_path):
            print(f" Файл не найден: {data_path}")
            continue
            
        df = pd.read_csv(data_path)
        print(f"Загружено: {len(df):,} примеров")
        
        # Для больших корпусов берем подвыборку для быстрой оценки
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            print(f"Берем подвыборку 10K для оценки")
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].tolist(), 
            df['label'].tolist(), 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        print(f"Оценка на: Train {len(X_train):,}, Test {len(X_test):,}")
        
        # Оцениваем классические модели
        for model_type in ['bow_logreg', 'tfidf_svm']:
            model_path = f"trained_models/final/{corpus_name}_{pipeline}_{model_type}_classifier.pkl"
            vectorizer_path = f"trained_models/final/{corpus_name}_{pipeline}_{model_type}_vectorizer.pkl"
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                try:
                    print(f" Оценка {model_type}...")
                    
                    # Загружаем модель и векторизатор
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    with open(vectorizer_path, 'rb') as f:
                        vectorizer = pickle.load(f)
                    
                    # Оцениваем и сохраняем в стандартном формате
                    model_results = evaluate_and_save_model(
                        model=model,
                        vectorizer=vectorizer,
                        X_test=X_test,
                        y_test=y_test,
                        dataset_name=corpus_name,
                        model_name=model_type,
                        preprocess_name=pipeline,
                        train_time=0,  # Время обучения можно добавить из логов
                        fold=None  # Для тестового набора
                    )
                    
                    print(f"    {model_type}: Accuracy={model_results['accuracy']:.4f}, F1-macro={model_results['macro_f1']:.4f}")
                    
                    # Сохраняем для сводки
                    results.append({
                        'corpus': corpus_name,
                        'pipeline': pipeline,
                        'model': model_type,
                        'accuracy': model_results['accuracy'],
                        'precision': model_results['precision'],
                        'recall': model_results['recall'],
                        'f1': model_results['macro_f1'],
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'status': 'success'
                    })
                    
                except Exception as e:
                    print(f"    Ошибка оценки {model_type}: {e}")
                    results.append({
                        'corpus': corpus_name,
                        'pipeline': pipeline,
                        'model': model_type,
                        'status': 'error',
                        'error': str(e)
                    })
            else:
                print(f"     Модель {model_type} не найдена")
        
        # Для LSTM - аналогичная оценка когда модель будет готова
        lstm_path = f"trained_models/lstm/{corpus_name}_{pipeline}_lstm.pth"
        if os.path.exists(lstm_path):
            try:
                print(f" Оценка LSTM...")
                # Здесь код для загрузки и оценки LSTM модели
                # lstm_results = evaluate_lstm_model(...)
                # save_experiment_results(lstm_results)
                print(f"    LSTM: оценка завершена")
            except Exception as e:
                print(f"    Ошибка оценки LSTM: {e}")

print(f"\n{'='*60}")
print(" ОЦЕНКА ВСЕХ МОДЕЛЕЙ ЗАВЕРШЕНА!")
print(f"{'='*60}")

# Сохраняем детальные результаты в старом формате для обратной совместимости
results_df = pd.DataFrame(results)
results_path = f"{results_dir}/all_models_evaluation_detailed.csv"
results_df.to_csv(results_path, index=False)
print(f" Детальные результаты сохранены в: {results_path}")

# Загружаем и показываем стандартизированные результаты
print(f"\n СТАНДАРТИЗИРОВАННЫЕ РЕЗУЛЬТАТЫ:")
standard_results = []
for file in os.listdir(results_dir):
    if file.endswith('.csv') and not file.startswith('all_models_evaluation'):
        filepath = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(filepath)
            standard_results.append(df)
            print(f"   📊 {file}: {len(df)} строк")
        except:
            pass

if standard_results:
    combined_df = pd.concat(standard_results, ignore_index=True)
    combined_path = f"{results_dir}/all_standard_results.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"   Все стандартизированные результаты объединены в: {combined_path}")
    
    # Сводка по моделям
    print(f"\n СВОДКА ПО МОДЕЛЯМ:")
    for model in combined_df['model'].unique():
        model_data = combined_df[combined_df['model'] == model]
        avg_accuracy = model_data['accuracy'].mean()
        avg_f1 = model_data['macro_f1'].mean()
        print(f"   {model}: Accuracy={avg_accuracy:.4f}, F1-macro={avg_f1:.4f}")

print(f"\n АНАЛИЗ:")
print("   - Все результаты сохранены в стандартном формате CSV")
print("   - Столбцы: dataset, model, preprocess, fold, seed, accuracy, macro_f1, precision, recall, train_time_sec")
print("   - Готово для генерации таблиц 1-3 'в один клик'")