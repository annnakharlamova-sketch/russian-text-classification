#!/usr/bin/env python3
"""
Скрипт для скачивания корпусов данных
"""

import os
import pandas as pd


def download_rusentiment():
    """Скачивание RuSentiment датасета"""
    print("Скачивание RuSentiment...")
    
    try:
        from datasets import load_dataset
        
        # Создаем папку
        os.makedirs('data/rusentiment', exist_ok=True)
        
        # Загружаем датасет
        dataset = load_dataset('MonoHime/ru_sentiment_dataset')
        
        # Сохраняем только существующие части
        print("Доступные разделы датасета:", list(dataset.keys()))
        
        if 'train' in dataset:
            dataset['train'].to_csv('data/rusentiment/train.csv', index=False)
            print(f"   Train сохранен: {len(dataset['train'])} примеров")
        
        if 'validation' in dataset:
            dataset['validation'].to_csv('data/rusentiment/validation.csv', index=False)
            print(f"   Validation сохранен: {len(dataset['validation'])} примеров")
            
        if 'test' in dataset:
            dataset['test'].to_csv('data/rusentiment/test.csv', index=False)
            print(f"   Test сохранен: {len(dataset['test'])} примеров")
        else:
            print("   Раздел 'test' не найден в датасете")
        
        # Покажем структуру данных
        if 'train' in dataset:
            print(f"   Колонки: {dataset['train'].column_names}")
        
        print("RuSentiment успешно скачан!")
        
    except ImportError:
        print("Библиотека 'datasets' не установлена")
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")


def check_datasets_installation():
    """Проверка установки библиотеки datasets"""
    try:
        import datasets
        print("Библиотека 'datasets' установлена")
        return True
    except ImportError:
        print("Библиотека 'datasets' не установлена")
        print("   Устанавливаем...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            print("Библиотека 'datasets' установлена успешно!")
            return True
        except Exception as e:
            print(f"Ошибка установки: {e}")
            return False


def main():
    """Главная функция"""
    print("Запуск скачивания данных для исследования")
    print("=" * 50)
    
    # Проверяем установку библиотек
    if check_datasets_installation():
        # Скачиваем RuSentiment
        download_rusentiment()
    
    print("=" * 50)
    print("Процесс скачивания завершен!")


if __name__ == "__main__":
    main()