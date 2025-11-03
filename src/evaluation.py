"""
Модуль для оценки моделей с полным протоколом
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import time
import os
from pathlib import Path  
from utils import get_global_seed


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.cv_results = []  
        
    def bootstrap_ci(self, y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
        """
        Расчет 95% доверительного интервала методом перцентильного бутстрэпа
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки  
            metric_fn: функция метрики
            n_bootstrap: количество бутстрэп выборок
            confidence: уровень доверия
            
        Returns:
            mean_score: среднее значение метрики
            ci: кортеж (lower, upper) доверительного интервала
        """
        # Фиксация seed для воспроизводимости бутстрэпа
        seed = get_global_seed()
        np.random.seed(seed)
        
        scores = []
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Бутстрэп выборка с заменой
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)
        
        # Перцентильный метод для доверительного интервала
        alpha = (1 - confidence) / 2
        lower = np.percentile(scores, 100 * alpha)
        upper = np.percentile(scores, 100 * (1 - alpha))
        mean_score = np.mean(scores)
        
        return mean_score, (lower, upper)
    
    def calculate_metrics(self, y_true, y_pred):
        """Расчет всех метрик качества"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def save_predictions(self, y_true, y_pred, y_pred_proba, dataset_name, model_name, preprocess_name):  
        """
        Сохранение предсказаний для построения графиков
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки
            y_pred_proba: вероятности предсказаний
            dataset_name: название датасета
            model_name: название модели
            preprocess_name: название пайплайна
        """
        predictions_df = pd.DataFrame({
            'dataset': [dataset_name] * len(y_true),
            'model': [model_name] * len(y_true),
            'preprocess': [preprocess_name] * len(y_true),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
        
        # Сохранение в файл
        predictions_path = Path('results/model_predictions.csv')
        if predictions_path.exists():
            existing_df = pd.read_csv(predictions_path)
            predictions_df = pd.concat([existing_df, predictions_df], ignore_index=True)
        
        predictions_df.to_csv(predictions_path, index=False)
        print(f" Предсказания сохранены: {predictions_path}")
    
    def cross_validate_model(self, model, X, y, dataset_name, model_name, preprocess_name):  
        """
        5-кратная стратифицированная кросс-валидация
        
        Args:
            model: модель для валидации
            X: признаки
            y: метки
            dataset_name: название датасета
            model_name: название модели
            preprocess_name: название пайплайна предобработки
            
        Returns:
            cv_results: результаты кросс-валидации
        """
        print(f" 5-кратная стратифицированная CV: {dataset_name} - {model_name} - {preprocess_name}")
        
        cv_config = self.config['evaluation']
        kf = StratifiedKFold(
            n_splits=cv_config['cv_folds'],
            shuffle=True,
            random_state=cv_config['random_state']
        )
        
        fold_results = []
        fold_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            start_time = time.time()
            
            X_train, X_val = X[train_idx], X[val_idx]  
            y_train, y_val = y[train_idx], y[val_idx]  
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_val)
            
            # Метрики
            metrics = self.calculate_metrics(y_val, y_pred)
            train_time = time.time() - start_time
            
            # Сохранение результатов фолда
            fold_result = {
                'dataset': dataset_name,
                'model': model_name,
                'preprocess': preprocess_name,
                'fold': fold + 1,
                'seed': get_global_seed(),
                'accuracy': metrics['accuracy'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'f1_macro': metrics['f1_macro'],
                'train_time_sec': train_time,
                'samples_count': len(X_val)
            }
            
            fold_results.append(fold_result)
            fold_times.append(train_time)
            
            print(f"   Fold {fold + 1}: F1-macro = {metrics['f1_macro']:.4f}, Time = {train_time:.2f}s")
        
        # Сохраняем детальные результаты
        self.cv_results.extend(fold_results)
        
        # Усреднение метрик по фолдам
        avg_metrics = self._aggregate_cv_results(fold_results)
        print(f"    CV Average: F1-macro = {avg_metrics['f1_macro_mean']:.4f} ± {avg_metrics['f1_macro_std']:.4f}")
        
        return fold_results, avg_metrics
    
    def _aggregate_cv_results(self, fold_results):
        """Агрегация результатов кросс-валидации"""
        df = pd.DataFrame(fold_results)
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        aggregated = {}
        for metric in metrics:
            values = df[metric]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_ci_lower'] = np.percentile(values, 2.5)
            aggregated[f'{metric}_ci_upper'] = np.percentile(values, 97.5)
        
        aggregated['total_train_time'] = df['train_time_sec'].sum()
        aggregated['avg_train_time'] = df['train_time_sec'].mean()
        
        return aggregated
    
    def evaluate_with_confidence_intervals(self, y_true, y_pred, dataset_name, model_name, preprocess_name):
        """
        Полная оценка модели с доверительными интервалами
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки
            dataset_name: название датасета
            model_name: название модели
            preprocess_name: название пайплайна предобработки
            
        Returns:
            result_dict: словарь с результатами
        """
        print(f" Оценка с доверительными интервалами: {model_name}")
        
        # Базовые метрики
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # 95% доверительные интервалы для F1-macro (перцентильный бутстрэп)
        f1_mean, f1_ci = self.bootstrap_ci(
            y_true, y_pred,
            lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            n_bootstrap=self.config['evaluation']['bootstrap_samples'],
            confidence=self.config['evaluation']['confidence_interval']
        )
        
        # Доверительные интервалы для других метрик
        accuracy_mean, accuracy_ci = self.bootstrap_ci(
            y_true, y_pred, accuracy_score,
            n_bootstrap=self.config['evaluation']['bootstrap_samples'],
            confidence=self.config['evaluation']['confidence_interval']
        )
        
        result = {
            'dataset': dataset_name,
            'model': model_name,
            'preprocess': preprocess_name,
            'seed': get_global_seed(),
            'accuracy': metrics['accuracy'],
            'accuracy_ci_lower': accuracy_ci[0],
            'accuracy_ci_upper': accuracy_ci[1],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'f1_macro_ci_lower': f1_ci[0],
            'f1_macro_ci_upper': f1_ci[1],
            'samples_count': len(y_true)
        }
        
        # Сохранение в общие результаты
        self.results.append(result)
        
        print(f"    F1-macro: {metrics['f1_macro']:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})")
        print(f"    Accuracy: {metrics['accuracy']:.4f} (95% CI: {accuracy_ci[0]:.4f}-{accuracy_ci[1]:.4f})")
        
        return result
    
    def save_results(self, output_dir="results"):
        """
        Сохранение результатов в CSV файлы
        
        Args:
            output_dir: директория для сохранения
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results:
            # Основные результаты
            results_df = pd.DataFrame(self.results)
            results_path = os.path.join(output_dir, "evaluation_results.csv")
            results_df.to_csv(results_path, index=False, encoding='utf-8')
            print(f" Результаты сохранены: {results_path}")
            print(f"   Всего записей: {len(results_df)}")
            
            # Детальные результаты по фолдам (если есть)
            if hasattr(self, 'cv_results') and self.cv_results:
                cv_df = pd.DataFrame(self.cv_results)
                cv_path = os.path.join(output_dir, "cv_detailed_results.csv")
                cv_df.to_csv(cv_path, index=False, encoding='utf-8')
                print(f" Детальные CV результаты сохранены: {cv_path}")
        
        return self.results
    
    def generate_summary_table(self):
        """
        Генерация сводной таблицы для статьи
        """
        if not self.results:
            print(" Нет результатов для генерации таблицы")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Группировка по датасетам и моделям
        summary = df.groupby(['dataset', 'model', 'preprocess']).agg({
            'accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'f1_macro_ci_lower': 'mean',
            'f1_macro_ci_upper': 'mean',
            'samples_count': 'mean'
        }).round(4)
        
        print(" СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
        print("=" * 80)
        print(summary)
        
        return summary
    
    def statistical_significance_test(self, y_true, pred1, pred2, metric='f1_macro'):
        """
        Статистический тест значимости различий между моделями
        """
        # Фиксация seed для воспроизводимости
        seed = get_global_seed()
        np.random.seed(seed)
        
        def metric_fn_f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        def metric_fn_accuracy(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        metric_fn = metric_fn_f1 if metric == 'f1_macro' else metric_fn_accuracy
        
        n_bootstrap = self.config['evaluation']['bootstrap_samples']
        scores1, scores2 = [], []
        
        n_samples = len(y_true)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score1 = metric_fn(y_true[indices], pred1[indices])
            score2 = metric_fn(y_true[indices], pred2[indices])
            scores1.append(score1)
            scores2.append(score2)
        
        # t-тест для парных выборок
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(np.array(scores1) - np.array(scores2)),
            'scores1_mean': np.mean(scores1),
            'scores2_mean': np.mean(scores2)
        }


def test_evaluation_protocol():
    """Тест протокола оценки"""
    print(" Тест протокола оценки...")
    
    test_config = {
        'evaluation': {
            'cv_folds': 5,
            'bootstrap_samples': 100,
            'confidence_interval': 0.95,
            'random_state': 42
        }
    }
    
    # Тестовые данные
    y_true = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    
    evaluator = Evaluator(test_config)
    
    # Тест бутстрэп CI
    mean_f1, f1_ci = evaluator.bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)
    )
    print(f" Bootstrap CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f}")
    
    # Тест полной оценки
    result = evaluator.evaluate_with_confidence_intervals(
        y_true, y_pred, 'test_dataset', 'test_model', 'P0'
    )
    print(f" Полная оценка: F1 = {result['f1_macro']:.4f}")
    
    # Тест сохранения
    evaluator.save_results('test_results')
    
    print("Протокол оценки работает корректно!")


if __name__ == "__main__":
    test_evaluation_protocol()