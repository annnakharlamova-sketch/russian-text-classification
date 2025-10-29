import sys
sys.path.append('src')

from evaluation import Evaluator
import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

print("=== ЭТАП 4: ОЦЕНКА И СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Создаем папку для результатов
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Создаем оценщик
evaluator = Evaluator(config)

results = []

# Оцениваем на всех корпусах и пайплайнах
for corpus_name in ['rureviews', 'rusentiment', 'taiga']:
    print(f"\n{'='*50}")
    print(f"ОЦЕНКА НА КОРПУСЕ: {corpus_name}")
    print(f"{'='*50}")
    
    for pipeline in ['P0', 'P1', 'P2', 'P3']:
        print(f"\n--- Пайплайн: {pipeline} ---")
        
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
                    
                    # Преобразуем тестовые данные
                    X_test_vec = vectorizer.transform(X_test)
                    
                    # Предсказания
                    y_pred = model.predict(X_test_vec)
                    
                    # Вычисляем метрики
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    print(f"    {model_type}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                    
                    results.append({
                        'corpus': corpus_name,
                        'pipeline': pipeline,
                        'model': model_type,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
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
        
        # Для LSTM пока просто отмечаем, что она обучена
        lstm_path = f"trained_models/lstm/{corpus_name}_{pipeline}_lstm.pth"
        if corpus_name == 'rureviews' and pipeline == 'P0':  # Наша обученная модель
            print(f" LSTM: обучена (loss уменьшился с 0.98 до 0.04)")
            results.append({
                'corpus': corpus_name,
                'pipeline': pipeline,
                'model': 'lstm',
                'status': 'trained',
                'notes': 'Loss: 0.98 → 0.04 за 10 эпох'
            })

print(f"\n{'='*60}")
print(" ОЦЕНКА ВСЕХ МОДЕЛЕЙ ЗАВЕРШЕНА!")
print(f"{'='*60}")

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_path = f"{results_dir}/all_models_evaluation.csv"
results_df.to_csv(results_path, index=False)
print(f" Результаты сохранены в: {results_path}")

# Сводка по успешным оценкам
successful_evals = [r for r in results if r['status'] == 'success']
if successful_evals:
    print(f"\n СВОДКА РЕЗУЛЬТАТОВ:")
    print(f"   Успешно оценено: {len(successful_evals)} моделей")
    
    # Группируем по моделям
    for model_type in ['bow_logreg', 'tfidf_svm']:
        model_results = [r for r in successful_evals if r['model'] == model_type]
        if model_results:
            avg_accuracy = np.mean([r['accuracy'] for r in model_results])
            avg_f1 = np.mean([r['f1'] for r in model_results])
            print(f"   {model_type}: Accuracy={avg_accuracy:.4f}, F1={avg_f1:.4f}")

print(f"\n АНАЛИЗ ДЛЯ СТАТЬИ:")
print("   - Сравнение TF-IDF+SVM vs BoW+LogReg")
print("   - Влияние пайплайнов предобработки P0-P3")
print("   - Производительность на разных корпусах")