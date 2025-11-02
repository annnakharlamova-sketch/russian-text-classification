"""
Скрипт для статистического анализа результатов
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evaluation import Evaluator

def perform_statistical_analysis():
    """Выполнение статистического анализа результатов"""
    results_path = "results/evaluation_results.csv"
    
    if not os.path.exists(results_path):
        print("Файл результатов не найден")
        return
    
    df = pd.read_csv(results_path)
    print(" СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # Сравнение моделей по датасетам
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        models = dataset_data['model'].unique()
        
        print(f"\n {dataset.upper()}:")
        
        if len(models) >= 2:
            # Попарное сравнение моделей
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    
                    data1 = dataset_data[dataset_data['model'] == model1]
                    data2 = dataset_data[dataset_data['model'] == model2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # t-тест для F1-macro
                        t_stat, p_value = stats.ttest_ind(
                            data1['f1_macro'], data2['f1_macro']
                        )
                        
                        significance = " СТАТИСТИЧЕСКИ ЗНАЧИМО" if p_value < 0.05 else " НЕ ЗНАЧИМО"
                        print(f"   {model1} vs {model2}: p-value = {p_value:.4f} {significance}")
    
    print(f"\n{'='*60}")
    print("Анализ завершен!")

if __name__ == "__main__":
    perform_statistical_analysis()