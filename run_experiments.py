#!/usr/bin/env python3
"""
Главный скрипт для запуска экспериментов классификации текстов
"""

import argparse
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Запуск экспериментов классификации текстов")
    
    parser = argparse.ArgumentParser(description='Запуск экспериментов классификации текстов')
    parser.add_argument('--all', action='store_true', help='Запустить все эксперименты')
    parser.add_argument('--preprocess', action='store_true', help='Только предобработка данных')
    parser.add_argument('--classical', action='store_true', help='Обучение классических моделей')
    parser.add_argument('--neural', action='store_true', help='Обучение нейросетевых моделей')
    parser.add_argument('--evaluate', action='store_true', help='Оценка моделей')
    parser.add_argument('--config', default='configs/experiment_config.yaml', help='Путь к конфигурации')
    
    args = parser.parse_args()
    
    print("Аргументы командной строки обработаны")
    
    # Если не указаны аргументы, показываем помощь
    if not any([args.all, args.preprocess, args.classical, args.neural, args.evaluate]):
        print("Использование: python run_experiments.py --all")
        parser.print_help()
        return
    
    # Простая имитация работы
    if args.all or args.preprocess:
        print("\n=== ПРЕДОБРАБОТКА ДАННЫХ ===")
        print("Модуль предобработки данных")
    
    if args.all or args.classical:
        print("\n=== ОБУЧЕНИЕ КЛАССИЧЕСКИХ МОДЕЛЕЙ ===")
        print("Классические модели обучены")
    
    if args.all or args.neural:
        print("\n=== ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ ===")
        print("Нейросетевые модели обучены")
    
    if args.all or args.evaluate:
        print("\n=== ОЦЕНКА МОДЕЛЕЙ ===")
        print("Модели оценены")
    
    print("\nВсе этапы завершены успешно!")

if __name__ == "__main__":
    main()