"""
Анализ влияния предобработки текстов
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class PreprocessingAnalyzer:
    def __init__(self, config):
        self.config = config
    
    def analyze_vocabulary_reduction(self, processed_data):
        """Анализ сокращения словаря после предобработки"""
        print(" АНАЛИЗ СОКРАЩЕНИЯ СЛОВАРЯ:")
        print("=" * 50)
        
        results = {}
        
        for corpus_name, pipelines_data in processed_data.items():
            print(f"\n Корпус: {corpus_name}")
            corpus_results = {}
            
            for pipeline_name, data in pipelines_data.items():
                texts = data['processed_text'].tolist()
                
                # Анализ словаря
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(texts)
                vocab_size = len(vectorizer.vocabulary_)
                
                # Уникальные токены
                all_tokens = ' '.join(texts).split()
                unique_tokens = len(set(all_tokens))
                total_tokens = len(all_tokens)
                
                corpus_results[pipeline_name] = {
                    'vocab_size': vocab_size,
                    'unique_tokens': unique_tokens,
                    'total_tokens': total_tokens,
                    'avg_tokens_per_doc': total_tokens / len(texts)
                }
                
                print(f"   {pipeline_name}:")
                print(f"      Словарь: {vocab_size} токенов")
                print(f"      Уникальные токены: {unique_tokens}")
                print(f"      Средняя длина: {total_tokens / len(texts):.1f} токенов")
            
            results[corpus_name] = corpus_results
        
        return results
    
    def plot_preprocessing_impact(self, analysis_results, output_dir="results/figures"):
        """Визуализация влияния предобработки"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for corpus_name, corpus_results in analysis_results.items():
            # График размера словаря
            pipelines = list(corpus_results.keys())
            vocab_sizes = [corpus_results[p]['vocab_size'] for p in pipelines]
            unique_tokens = [corpus_results[p]['unique_tokens'] for p in pipelines]
            
            plt.figure(figsize=(12, 8))
            
            # График 1: Размер словаря
            plt.subplot(2, 2, 1)
            bars = plt.bar(pipelines, vocab_sizes, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            plt.title(f'Размер словаря - {corpus_name}')
            plt.ylabel('Количество токенов')
            plt.xticks(rotation=45)
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars, vocab_sizes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value}', ha='center', va='bottom')
            
            # График 2: Уникальные токены
            plt.subplot(2, 2, 2)
            bars = plt.bar(pipelines, unique_tokens, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            plt.title(f'Уникальные токены - {corpus_name}')
            plt.ylabel('Количество токенов')
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, unique_tokens):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value}', ha='center', va='bottom')
            
            # График 3: Средняя длина документа
            avg_lengths = [corpus_results[p]['avg_tokens_per_doc'] for p in pipelines]
            plt.subplot(2, 2, 3)
            bars = plt.bar(pipelines, avg_lengths, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            plt.title(f'Средняя длина документа - {corpus_name}')
            plt.ylabel('Токенов на документ')
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, avg_lengths):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            # График 4: Сокращение словаря (%)
            base_vocab = vocab_sizes[0]  # P0 как базовый
            reduction_pct = [((base_vocab - size) / base_vocab * 100) for size in vocab_sizes]
            
            plt.subplot(2, 2, 4)
            bars = plt.bar(pipelines, reduction_pct, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            plt.title(f'Сокращение словаря - {corpus_name}')
            plt.ylabel('Сокращение (%)')
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, reduction_pct):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/preprocessing_impact_{corpus_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f" Сохранен график: {output_dir}/preprocessing_impact_{corpus_name}.png")
    
    def create_preprocessing_table(self, analysis_results, output_dir="results/tables"):
        """Создание таблицы влияния предобработки"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        table_data = []
        
        for corpus_name, corpus_results in analysis_results.items():
            for pipeline_name, metrics in corpus_results.items():
                table_data.append({
                    'Корпус': corpus_name,
                    'Пайплайн': pipeline_name,
                    'Размер словаря': metrics['vocab_size'],
                    'Уникальные токены': metrics['unique_tokens'],
                    'Всего токенов': metrics['total_tokens'],
                    'Ср. длина документа': f"{metrics['avg_tokens_per_doc']:.1f}"
                })
        
        df = pd.DataFrame(table_data)
        
        # Сохраняем в CSV
        csv_path = f'{output_dir}/preprocessing_analysis.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Красивый вывод в консоль
        print("\n ТАБЛИЦА ВЛИЯНИЯ ПРЕДОБРАБОТКИ:")
        print("=" * 80)
        print(df.to_string(index=False))
        
        print(f" Сохранена таблица: {csv_path}")
        
        return df

def test_analyzer():
    """Тест анализатора предобработки"""
    print(" Тестирование анализатора предобработки...")
    
    # Тестовые данные
    test_data = {
        'test_corpus': {
            'P0': pd.DataFrame({
                'processed_text': ['отличный фильм очень понравилось', 'ужасное кино скучно']
            }),
            'P1': pd.DataFrame({
                'processed_text': ['отличный фильм понравилось', 'ужасное кино скучно']
            }),
            'P2': pd.DataFrame({
                'processed_text': ['отличн фильм понравилось', 'ужасн кин скучно']
            }),
            'P3': pd.DataFrame({
                'processed_text': ['отличный фильм понравиться', 'ужасный кино скучный']
            })
        }
    }
    
    test_config = {
        'preprocessing': {
            'pipelines': {
                'P0': ['clean_text', 'lowercase'],
                'P1': ['clean_text', 'lowercase', 'remove_stopwords'],
                'P2': ['clean_text', 'lowercase', 'remove_stopwords', 'stemming'],
                'P3': ['clean_text', 'lowercase', 'remove_stopwords', 'lemmatization']
            }
        }
    }
    
    analyzer = PreprocessingAnalyzer(test_config)
    results = analyzer.analyze_vocabulary_reduction(test_data)
    table = analyzer.create_preprocessing_table(results)
    
    print(" Тест анализатора завершен успешно!")

if __name__ == "__main__":
    test_analyzer()