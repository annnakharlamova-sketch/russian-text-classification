"""
Модуль для оценки моделей и статистических тестов
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold


class Evaluator:
    def __init__(self, config):
        self.config = config
    
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
    
    def bootstrap_confidence_interval(self, y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
        """Расчет доверительного интервала методом бутстрэпа"""
        scores = []
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Бутстрэп выборка с заменой
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)
        
        # Расчет доверительного интервала
        alpha = (1 - confidence) / 2
        lower = np.percentile(scores, 100 * alpha)
        upper = np.percentile(scores, 100 * (1 - alpha))
        mean_score = np.mean(scores)
        
        return mean_score, (lower, upper)
    
    def statistical_significance_test(self, y_true, pred1, pred2, metric='f1_macro'):
        """Статистический тест значимости различий между моделями"""
        # Преобразование в numpy arrays
        y_true = np.array(y_true)
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        
        # Расчет метрик для каждой бутстрэп выборки
        n_bootstrap = self.config['evaluation'].get('bootstrap_samples', 100)
        scores1, scores2 = [], []
        
        def metric_fn_f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        def metric_fn_accuracy(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        metric_fn = metric_fn_f1 if metric == 'f1_macro' else metric_fn_accuracy
        
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
            'mean_diff': np.mean(np.array(scores1) - np.array(scores2))
        }
    
    def cross_validate_model(self, model, X, y, cv_folds=5):
        """Кросс-валидация модели"""
        kf = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=self.config['evaluation']['random_state']
        )
        
        cv_scores = cross_val_score(
            model, X, y, 
            cv=kf, 
            scoring='f1_macro'
        )
        
        return {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'all_scores': cv_scores,
            'confidence_interval': (
                cv_scores.mean() - 1.96 * cv_scores.std() / np.sqrt(len(cv_scores)),
                cv_scores.mean() + 1.96 * cv_scores.std() / np.sqrt(len(cv_scores))
            )
        }
    
    def evaluate_classical_model(self, model, vectorizer, X_test, y_test, model_name):
        """Оценка классической модели с полной статистикой"""
        print(f"📊 Полная оценка {model_name}...")
        
        # Преобразование тестовых данных
        X_test_vec = vectorizer.transform(X_test)
        
        # Предсказания
        y_pred = model.predict(X_test_vec)
        
        # Базовые метрики
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Доверительные интервалы для F1
        f1_mean, f1_ci = self.bootstrap_confidence_interval(
            y_test, y_pred, 
            lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            n_bootstrap=self.config['evaluation'].get('bootstrap_samples', 100)
        )
        
        metrics.update({
            'f1_ci_lower': f1_ci[0],
            'f1_ci_upper': f1_ci[1],
            'f1_mean_bootstrap': f1_mean
        })
        
        print(f"   ✅ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   ✅ F1-macro:  {metrics['f1_macro']:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})")
        
        return metrics, y_pred
    
    def compare_models(self, y_true, predictions_dict):
        """Сравнение нескольких моделей между собой"""
        print("\n🔍 СРАВНЕНИЕ МОДЕЛЕЙ:")
        print("=" * 60)
        
        model_names = list(predictions_dict.keys())
        results = {}
        
        # Расчет метрик для всех моделей
        for name, y_pred in predictions_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            f1_mean, f1_ci = self.bootstrap_confidence_interval(
                y_true, y_pred,
                lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
                n_bootstrap=100
            )
            
            results[name] = {
                'metrics': metrics,
                'f1_ci': f1_ci,
                'predictions': y_pred
            }
            
            print(f"📈 {name.upper():<15}")
            print(f"   F1-macro: {metrics['f1_macro']:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print()
        
        # Статистические тесты попарно
        if len(model_names) >= 2:
            print("📊 СТАТИСТИЧЕСКИЕ ТЕСТЫ (F1-macro):")
            print("-" * 40)
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    test_result = self.statistical_significance_test(
                        y_true, 
                        predictions_dict[model1], 
                        predictions_dict[model2]
                    )
                    
                    significance = "✅ СТАТИСТИЧЕСКИ ЗНАЧИМО" if test_result['significant'] else "❌ НЕ ЗНАЧИМО"
                    print(f"   {model1} vs {model2}:")
                    print(f"      p-value: {test_result['p_value']:.4f} {significance}")
                    print(f"      Разница: {test_result['mean_diff']:.4f}")
                    print()
        
        return results
    
    def create_results_table(self, results_dict):
        """Создание таблицы результатов для статьи"""
        print("\n📋 ТАБЛИЦА РЕЗУЛЬТАТОВ:")
        print("=" * 80)
        
        table_data = []
        for model_name, result in results_dict.items():
            metrics = result['metrics']
            f1_ci = result.get('f1_ci', (0, 0))
            
            table_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
                'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
                'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
                'F1 95% CI': f"{f1_ci[0]:.4f}-{f1_ci[1]:.4f}"
            })
        
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        return df


def test_evaluator():
    """Тест системы оценки"""
    print("🧪 Тестирование системы оценки...")
    
    test_config = {
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'confidence_interval': 0.95,
            'bootstrap_samples': 100,
            'random_state': 42
        }
    }
    
    # Тестовые данные
    y_true = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    y_pred1 = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # Идеальные предсказания
    y_pred2 = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  # Одна ошибка
    
    evaluator = Evaluator(test_config)
    
    # Тест базовых метрик
    metrics = evaluator.calculate_metrics(y_true, y_pred1)
    print(f"✅ Базовые метрики: F1 = {metrics['f1_macro']:.4f}")
    
    # Тест доверительных интервалов
    mean_f1, ci = evaluator.bootstrap_confidence_interval(y_true, y_pred1, 
        lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0))
    print(f"✅ Доверительный интервал: {ci[0]:.4f}-{ci[1]:.4f}")
    
    # Тест статистической значимости
    test_result = evaluator.statistical_significance_test(y_true, y_pred1, y_pred2)
    print(f"✅ Статистический тест: p-value = {test_result['p_value']:.4f}")
    
    # Тест сравнения моделей
    predictions_dict = {
        'Perfect_Model': y_pred1,
        'Good_Model': y_pred2
    }
    comparison = evaluator.compare_models(y_true, predictions_dict)
    
    print("🧪 Тест системы оценки завершен успешно!")


if __name__ == "__main__":
    test_evaluator()