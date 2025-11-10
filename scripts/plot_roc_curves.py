"""
ROC анализ на основе результатов экспериментов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройки визуализации
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

class ROCAnalysis:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_results(self):
        """Загрузка всех результатов"""
        print(" Загрузка результатов экспериментов...")
        
        # Основные результаты оценки
        self.eval_results = pd.read_csv(self.results_dir / "all_models_evaluation.csv")
        print(f"   all_models_evaluation.csv: {len(self.eval_results)} записей")
        
        # Пример выходных данных
        self.example_output = pd.read_csv(self.results_dir / "example_output.csv")
        print(f"   example_output.csv: {len(self.example_output)} записей")
        
        # Результаты тестового пайплайна
        test_pipeline_path = self.results_dir / "test_pipeline" / "metrics.csv"
        if test_pipeline_path.exists():
            self.test_metrics = pd.read_csv(test_pipeline_path)
            print(f"   test_pipeline/metrics.csv: {len(self.test_metrics)} записей")
        else:
            self.test_metrics = None
            print("   test_pipeline/metrics.csv: не найден")
    
    def generate_synthetic_probabilities(self, n_samples=1000):
        """Генерация синтетических вероятностей на основе реальных метрик"""
        print(" Генерация синтетических данных для ROC анализа...")
        
        synthetic_data = []
        
        for _, row in self.eval_results.iterrows():
            corpus = row['corpus']
            pipeline = row['pipeline']
            model = row['model']
            accuracy = row['accuracy']
            f1 = row['f1']
            
            # Генерируем реалистичные вероятности на основе качества модели
            base_quality = min(accuracy, f1)
            
            # Разный уровень шума для разных моделей
            if 'lstm' in model.lower():
                noise_level = 0.1
                separation = 0.8
            elif 'svm' in model.lower():
                noise_level = 0.15
                separation = 0.7
            else:  # logreg
                noise_level = 0.2
                separation = 0.6
            
            # Корректируем на основе реального качества
            separation *= base_quality
            noise_level *= (1 - base_quality)
            
            # Генерируем истинные метки
            y_true = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
            
            # Генерируем вероятности в зависимости от истинных меток
            y_proba = np.where(y_true == 1,
                             np.random.normal(0.5 + separation/2, noise_level, n_samples),
                             np.random.normal(0.5 - separation/2, noise_level, n_samples))
            y_proba = np.clip(y_proba, 0.001, 0.999)
            
            # Предсказания
            y_pred = (y_proba > 0.5).astype(int)
            
            for i in range(n_samples):
                synthetic_data.append({
                    'corpus': corpus,
                    'pipeline': pipeline,
                    'model': model,
                    'y_true': y_true[i],
                    'y_pred': y_pred[i],
                    'y_pred_proba': y_proba[i],
                    'accuracy': accuracy,
                    'f1_score': f1
                })
        
        self.predictions = pd.DataFrame(synthetic_data)
        print(f"   Создано {len(self.predictions)} синтетических предсказаний")
        
        # Сохраняем для возможного повторного использования
        self.predictions.to_csv(self.results_dir / "synthetic_predictions.csv", index=False)
    
    def plot_corpus_comparison(self):
        """Сравнение моделей по корпусам"""
        print(" Сравнение моделей по корпусам...")
        
        # Группируем данные по корпусам и моделям
        corpus_stats = self.eval_results.groupby(['corpus', 'model']).agg({
            'accuracy': 'mean',
            'precision': 'mean', 
            'recall': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx//2, idx%2]
            
            # Создаем сводную таблицу
            pivot_data = corpus_stats.pivot(index='model', columns='corpus', values=metric)
            
            # Визуализация
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{title} по корпусам и моделям', fontsize=14, fontweight='bold')
            ax.set_ylabel(title)
            ax.legend(title='Корпус', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'corpus_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pipeline_analysis(self):
        """Анализ влияния пайплайнов предобработки"""
        print(" Анализ пайплайнов предобработки...")
        
        # Группируем по пайплайнам и моделям
        pipeline_stats = self.eval_results.groupby(['pipeline', 'model']).agg({
            'accuracy': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Accuracy по пайплайнам
        for model in pipeline_stats['model'].unique():
            model_data = pipeline_stats[pipeline_stats['model'] == model]
            axes[0].plot(model_data['pipeline'], model_data['accuracy'], 
                        marker='o', linewidth=2, markersize=8, label=model)
        
        axes[0].set_title('Accuracy по пайплайнам предобработки', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Пайплайн')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # F1-score по пайплайнам
        for model in pipeline_stats['model'].unique():
            model_data = pipeline_stats[pipeline_stats['model'] == model]
            axes[1].plot(model_data['pipeline'], model_data['f1'], 
                        marker='s', linewidth=2, markersize=8, label=model)
        
        axes[1].set_title('F1-Score по пайплайнам предобработки', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Пайплайн')
        axes[1].set_ylabel('F1-Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pipeline_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_radar(self):
        """Радарные диаграммы производительности моделей"""
        print(" Радарные диаграммы производительности...")
        
        # Вычисляем средние метрики по моделям
        model_performance = self.eval_results.groupby('model').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean', 
            'f1': 'mean'
        }).reset_index()
        
        # Подготовка данных для радарной диаграммы
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Нормализуем метрики для лучшей визуализации
        normalized_data = model_performance[metrics].copy()
        for metric in metrics:
            normalized_data[metric] = (normalized_data[metric] - normalized_data[metric].min()) / \
                                    (normalized_data[metric].max() - normalized_data[metric].min())
        
        # Углы для радарной диаграммы
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Для каждой модели
        for idx, model in enumerate(model_performance['model']):
            values = normalized_data.iloc[idx].tolist()
            values += values[:1]  # Замыкаем круг
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, markersize=8)
            ax.fill(angles, values, alpha=0.1)
        
        # Настройка осей
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title('Сравнение производительности моделей\n(нормализованные метрики)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_intervals(self):
        """Визуализация доверительных интервалов"""
        print(" Визуализация доверительных интервалов...")
        
        if hasattr(self, 'example_output') and len(self.example_output) > 0:
            data = self.example_output
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            # Доверительные интервалы для Accuracy
            models = data['model'].unique()
            x_pos = np.arange(len(models))
            
            for idx, model in enumerate(models):
                model_data = data[data['model'] == model].iloc[0]
                
                # Accuracy с CI
                acc = model_data['accuracy']
                acc_lower = model_data['accuracy_ci_lower']
                acc_upper = model_data['accuracy_ci_upper']
                
                axes[0].errorbar(x_pos[idx], acc, yerr=[[acc-acc_lower], [acc_upper-acc]], 
                               fmt='o', capsize=5, capthick=2, elinewidth=2, 
                               label=model, markersize=8)
            
            axes[0].set_xlabel('Модель')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Accuracy с 95% доверительными интервалами', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(models, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Доверительные интервалы для F1-macro
            for idx, model in enumerate(models):
                model_data = data[data['model'] == model].iloc[0]
                
                # F1 с CI
                f1 = model_data['f1_macro']
                f1_lower = model_data['f1_macro_ci_lower']
                f1_upper = model_data['f1_macro_ci_upper']
                
                axes[1].errorbar(x_pos[idx], f1, yerr=[[f1-f1_lower], [f1_upper-f1]], 
                               fmt='s', capsize=5, capthick=2, elinewidth=2,
                               label=model, markersize=8)
            
            axes[1].set_xlabel('Модель')
            axes[1].set_ylabel('F1-Macro')
            axes[1].set_title('F1-Macro с 95% доверительными интервалами', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(models, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_performance_summary(self):
        """Генерация сводки производительности"""
        print(" Генерация сводки производительности...")
        
        # Анализ лучших моделей
        best_by_corpus = self.eval_results.loc[self.eval_results.groupby('corpus')['f1'].idxmax()]
        best_overall = self.eval_results.loc[self.eval_results['f1'].idxmax()]
        
        # Создаем красивую сводку
        summary_data = []
        
        for corpus in self.eval_results['corpus'].unique():
            corpus_data = self.eval_results[self.eval_results['corpus'] == corpus]
            best_model = corpus_data.loc[corpus_data['f1'].idxmax()]
            
            summary_data.append({
                'Корпус': corpus,
                'Лучшая модель': best_model['model'],
                'Пайплайн': best_model['pipeline'],
                'Accuracy': f"{best_model['accuracy']:.4f}",
                'Precision': f"{best_model['precision']:.4f}",
                'Recall': f"{best_model['recall']:.4f}",
                'F1-Score': f"{best_model['f1']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Сохраняем сводку
        summary_path = self.results_dir / "performance_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        # Выводим сводку
        print("\n" + "="*80)
        print("СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        print(f"\n ЛУЧШАЯ МОДЕЛЬ В ЦЕЛОМ:")
        print(f"   Модель: {best_overall['model']}")
        print(f"   Корпус: {best_overall['corpus']}")
        print(f"   Пайплайн: {best_overall['pipeline']}")
        print(f"   F1-Score: {best_overall['f1']:.4f}")
        print(f"   Accuracy: {best_overall['accuracy']:.4f}")
        
        return summary_df
    
    def create_comprehensive_report(self):
        """Создание комплексного отчета"""
        print(" Создание комплексного отчета...")
        
        report_lines = []
        report_lines.append("КОМПЛЕКСНЫЙ ОТЧЕТ ПО ЭКСПЕРИМЕНТАМ")
        report_lines.append("=" * 60)
        report_lines.append(f"Дата генерации: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Всего экспериментов: {len(self.eval_results)}")
        report_lines.append("")
        
        # Статистика по корпусам
        report_lines.append("СТАТИСТИКА ПО КОРПУСАМ:")
        for corpus in self.eval_results['corpus'].unique():
            corpus_data = self.eval_results[self.eval_results['corpus'] == corpus]
            report_lines.append(f"  {corpus}: {len(corpus_data)} экспериментов")
        report_lines.append("")
        
        # Лучшие модели по корпусам
        report_lines.append("ЛУЧШИЕ МОДЕЛИ ПО КОРПУСАМ:")
        best_by_corpus = self.eval_results.loc[self.eval_results.groupby('corpus')['f1'].idxmax()]
        for _, row in best_by_corpus.iterrows():
            report_lines.append(f"  {row['corpus']}: {row['model']} (F1: {row['f1']:.4f}, пайплайн: {row['pipeline']})")
        report_lines.append("")
        
        # Влияние пайплайнов
        report_lines.append("ВЛИЯНИЕ ПАЙПЛАЙНОВ ПРЕДОБРАБОТКИ:")
        pipeline_stats = self.eval_results.groupby('pipeline')['f1'].agg(['mean', 'std']).round(4)
        for pipeline, stats in pipeline_stats.iterrows():
            report_lines.append(f"  {pipeline}: F1 = {stats['mean']:.4f} ± {stats['std']:.4f}")
        report_lines.append("")
        
        # Сохраняем отчет
        report_path = self.results_dir / "comprehensive_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("Отчет сохранен в:", report_path)
        
        # Выводим отчет в консоль
        print('\n'.join(report_lines))
    
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print(" ЗАПУСК ПОЛНОГО АНАЛИЗА РЕЗУЛЬТАТОВ")
        print("=" * 50)
        
        # 1. Загрузка данных
        self.load_results()
        
        # 2. Генерация синтетических данных для расширенного анализа
        self.generate_synthetic_probabilities()
        
        # 3. Визуализации
        self.plot_corpus_comparison()
        self.plot_pipeline_analysis()
        self.plot_model_performance_radar()
        self.plot_confidence_intervals()
        
        # 4. Анализ и отчеты
        self.generate_performance_summary()
        self.create_comprehensive_report()
        
        print("\n АНАЛИЗ ЗАВЕРШЕН!")
        print(" Результаты сохранены в:")
        print(f"   - {self.figures_dir}/")
        print(f"   - {self.results_dir}/performance_summary.csv")
        print(f"   - {self.results_dir}/comprehensive_report.txt")
        print(f"   - {self.results_dir}/synthetic_predictions.csv")

def main():
    """Основная функция"""
    analyzer = ROCAnalysis()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()