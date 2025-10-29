import sys
sys.path.append('src')

import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle
import time

print("=== ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ===")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Создаем папку для моделей
model_dir = "trained_models/final"
os.makedirs(model_dir, exist_ok=True)

# Обучаем с оптимизированными настройками для больших корпусов
for corpus_name in ['rusentiment', 'taiga']:
    print(f"\n{'='*50}")
    print(f"ОБУЧЕНИЕ НА КОРПУСЕ: {corpus_name}")
    print(f"{'='*50}")
    
    for pipeline in ['P0', 'P1', 'P2', 'P3']:
        print(f"\n--- Пайплайн: {pipeline} ---")
        
        # Загружаем данные
        data_path = f"processed_data/{corpus_name}/{pipeline}.csv"
        if not os.path.exists(data_path):
            print(f" Файл не найден: {data_path}")
            continue
            
        df = pd.read_csv(data_path)
        print(f"Загружено: {len(df):,} примеров")
        
        # Проверяем классы
        unique_labels = df['label'].unique()
        print(f"Уникальные метки: {unique_labels}")
        
        if len(unique_labels) < 2:
            print(f" Недостаточно классов для обучения: {unique_labels}")
            continue
            
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].tolist(), 
            df['label'].tolist(), 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Адаптивные настройки в зависимости от размера данных
        if len(X_train) > 100000:  # Большие корпуса
            bow_max_features = 8000
            tfidf_max_features = 15000
        else:  # Средние корпуса
            bow_max_features = 10000
            tfidf_max_features = 20000
        
        # Обучаем BoW + Logistic Regression
        try:
            print(" Обучение BoW + Logistic Regression...")
            start_time = time.time()
            
            bow_vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                max_features=bow_max_features,
                min_df=5
            )
            
            X_train_bow = bow_vectorizer.fit_transform(X_train)
            print(f"   Размерность признаков: {X_train_bow.shape}")
            
            logreg = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            
            logreg.fit(X_train_bow, y_train)
            
            model_prefix = f"{model_dir}/{corpus_name}_{pipeline}_bow_logreg"
            with open(f"{model_prefix}_vectorizer.pkl", 'wb') as f:
                pickle.dump(bow_vectorizer, f)
            with open(f"{model_prefix}_classifier.pkl", 'wb') as f:
                pickle.dump(logreg, f)
            
            training_time = time.time() - start_time
            print(f" BoW+LogReg обучена за {training_time:.1f} сек")
            
        except Exception as e:
            print(f" Ошибка BoW+LogReg: {e}")
        
        # Обучаем TF-IDF + SVM с оптимизацией памяти
        try:
            print(" Обучение TF-IDF + SVM (оптимизированная версия)...")
            start_time = time.time()
            
            # Используем разреженные матрицы и уменьшенные max_features
            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=tfidf_max_features,  # Уменьшено для экономии памяти
                min_df=5,  # Увеличено min_df для уменьшения словаря
                sublinear_tf=True,
                use_idf=True,
                norm='l2'
            )
            
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            print(f"   Размерность признаков: {X_train_tfidf.shape}")
            
            svm = LinearSVC(
                C=1.0,
                random_state=42,
                max_iter=1000,
                dual=False  # Используем primal форму для экономии памяти
            )
            
            svm.fit(X_train_tfidf, y_train)
            
            model_prefix = f"{model_dir}/{corpus_name}_{pipeline}_tfidf_svm"
            with open(f"{model_prefix}_vectorizer.pkl", 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
            with open(f"{model_prefix}_classifier.pkl", 'wb') as f:
                pickle.dump(svm, f)
            
            training_time = time.time() - start_time
            print(f" TF-IDF+SVM обучена за {training_time:.1f} сек")
            
        except Exception as e:
            print(f" Ошибка TF-IDF+SVM: {e}")
            print("   Пробуем еще более оптимизированную версию...")
            
            try:
                # Еще более агрессивная оптимизация
                print(" Обучение TF-IDF + SVM (супер-оптимизированная)...")
                start_time = time.time()
                
                tfidf_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 1),  # Только униграммы
                    max_features=10000,  # Еще меньше признаков
                    min_df=10,  # Еще более строгий min_df
                    sublinear_tf=True,
                    use_idf=True,
                    norm='l2'
                )
                
                X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
                print(f"   Размерность признаков: {X_train_tfidf.shape}")
                
                svm = LinearSVC(
                    C=1.0,
                    random_state=42,
                    max_iter=1000,
                    dual=False
                )
                
                svm.fit(X_train_tfidf, y_train)
                
                model_prefix = f"{model_dir}/{corpus_name}_{pipeline}_tfidf_svm_optimized"
                with open(f"{model_prefix}_vectorizer.pkl", 'wb') as f:
                    pickle.dump(tfidf_vectorizer, f)
                with open(f"{model_prefix}_classifier.pkl", 'wb') as f:
                    pickle.dump(svm, f)
                
                training_time = time.time() - start_time
                print(f" TF-IDF+SVM (оптимизированная) обучена за {training_time:.1f} сек")
                
            except Exception as e2:
                print(f" Ошибка даже в оптимизированной версии: {e2}")

print(f"\n ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")

# Проверим итоговые модели
print(f"\n ИТОГОВАЯ ПРОВЕРКА МОДЕЛЕЙ:")
if os.path.exists(model_dir):
    models = os.listdir(model_dir)
    print(f"Всего файлов моделей: {len(models)}")
    
    # Сгруппируем по корпусам
    for corpus in ['rureviews', 'rusentiment', 'taiga']:
        corpus_models = [m for m in models if corpus in m]
        if corpus_models:
            print(f"  {corpus}: {len(corpus_models)} файлов")