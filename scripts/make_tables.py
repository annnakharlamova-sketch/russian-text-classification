"""
Генерация таблиц для научной статьи (Таблицы 1-3)
"""

import pandas as pd
import numpy as np
import os

def load_results():
    """Загрузка результатов экспериментов"""
    results_path = "results/all_models_evaluation.csv"
    if not os.path.exists(results_path):
        print(" Файл с результатами не найден. Сначала запустите эксперименты.")
        return None
    
    df = pd.read_csv(results_path)
    print(f" Загружены результаты: {len(df)} записей")
    return df

def create_table1_model_comparison(df):
    """Таблица 1: Сравнение моделей (Accuracy, F1-macro)"""
    print("Создание Таблицы 1: Сравнение моделей...")
    
    # Группируем по моделям
    table1 = df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'count'],
        'f1': ['mean', 'std']
    }).round(4)
    
    # Переименовываем колонки для читаемости
    table1.columns = ['Accuracy_Mean', 'Accuracy_Std', 'N_Experiments', 'F1_Mean', 'F1_Std']
    table1 = table1.reset_index()
    
    # Сохраняем
    os.makedirs('results/tables', exist_ok=True)
    table1.to_csv('results/tables/table1_model_comparison.csv', index=False)
    
    print(" Таблица 1 сохранена: results/tables/table1_model_comparison.csv")
    return table1

def create_table2_preprocessing_impact(df):
    """Таблица 2: Влияние предобработки на точность"""
    print("Создание Таблицы 2: Влияние предобработки...")
    
    # Группируем по пайплайнам предобработки
    table2 = df.groupby(['pipeline', 'model']).agg({
        'accuracy': 'mean',
        'f1': 'mean'
    }).round(4).reset_index()
    
    # Pivot таблица для удобства
    table2_pivot = table2.pivot(index='pipeline', columns='model', values=['accuracy', 'f1'])
    table2_pivot = table2_pivot.round(4)
    
    # Сохраняем
    table2.to_csv('results/tables/table2_preprocessing_impact.csv', index=False)
    table2_pivot.to_csv('results/tables/table2_preprocessing_impact_pivot.csv')
    
    print(" Таблица 2 сохранена: results/tables/table2_preprocessing_impact.csv")
    return table2

def create_table3_corpus_comparison(df):
    """Таблица 3: Сравнение производительности по корпусам"""
    print("Создание Таблицы 3: Сравнение корпусов...")
    
    # Группируем по корпусам и моделям
    table3 = df.groupby(['corpus', 'model']).agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'pipeline': 'count'
    }).round(4)
    
    table3.columns = ['Accuracy_Mean', 'Accuracy_Std', 'F1_Mean', 'F1_Std', 'N_Experiments']
    table3 = table3.reset_index()
    
    # Сохраняем
    table3.to_csv('results/tables/table3_corpus_comparison.csv', index=False)
    
    print(" Таблица 3 сохранена: results/tables/table3_corpus_comparison.csv")
    return table3

def create_article_summary(df):
    """Создание краткой сводки для статьи"""
    print("Создание сводки для статьи...")
    
    # Лучшие результаты по каждому корпусу
    best_results = df.loc[df.groupby('corpus')['accuracy'].idxmax()]
    
    summary = f"""
КРАТКАЯ СВОДКА ДЛЯ СТАТЬИ:
==========================

ОБЩИЕ РЕЗУЛЬТАТЫ:
- Всего экспериментов: {len(df)}
- Средняя точность: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}
- Средний F1-score: {df['f1'].mean():.3f} ± {df['f1'].std():.3f}

ЛУЧШИЕ РЕЗУЛЬТАТЫ ПО КОРПУСАМ:
{best_results[['corpus', 'model', 'pipeline', 'accuracy', 'f1']].to_string(index=False)}

ТОП-3 МОДЕЛИ:
{df.groupby('model')['accuracy'].mean().sort_values(ascending=False).head(3).to_string()}

ВЛИЯНИЕ ПРЕДОБРАБОТКИ:
{df.groupby('pipeline')['accuracy'].mean().sort_values(ascending=False).to_string()}
"""
    
    with open('results/tables/article_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(" Сводка сохранена: results/tables/article_summary.txt")
    return summary

def main():
    """Главная функция"""
    print(" ГЕНЕРАЦИЯ ТАБЛИЦ ДЛЯ НАУЧНОЙ СТАТЬИ")
    print("=" * 50)
    
    # Загружаем результаты
    df = load_results()
    if df is None:
        return
    
    # Создаем таблицы
    table1 = create_table1_model_comparison(df)
    table2 = create_table2_preprocessing_impact(df) 
    table3 = create_table3_corpus_comparison(df)
    summary = create_article_summary(df)
    
    # Выводим превью
    print("\n" + "=" * 50)
    print("ПРЕВЬЮ ТАБЛИЦЫ 1 (Сравнение моделей):")
    print(table1.head().to_string(index=False))
    
    print(f"\n ВСЕ ТАБЛИЦЫ СОЗДАНЫ!")
    print("   - results/tables/table1_model_comparison.csv")
    print("   - results/tables/table2_preprocessing_impact.csv") 
    print("   - results/tables/table3_corpus_comparison.csv")
    print("   - results/tables/article_summary.txt")

if __name__ == "__main__":
    main()