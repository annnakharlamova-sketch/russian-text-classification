"""
Вспомогательные функции
"""

import os
import yaml


def load_config(config_path):
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(directory):
    """Создание директории, если она не существует"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Создана директория: {directory}")
    return directory


def print_config_summary(config):
    """Вывод краткой информации о конфигурации"""
    print("Конфигурация эксперимента:")
    print(f"   Корпусы: {list(config['data']['corpora'].keys())}")
    print(f"   Пайплайны: {list(config['preprocessing']['pipelines'].keys())}")
    print(f"   Модели: {list(config['models']['classical'].keys())} + LSTM")
    print(f"   Метрики: {config['evaluation']['metrics']}")


def test_utils():
    """Тестовая функция для проверки модуля"""
    print("Тест вспомогательных функций:")
    
    # Тест создания директории
    test_dir = ensure_dir('test_directory')
    print(f"Директория создана: {test_dir}")
    
    # Удалим тестовую директорию
    import shutil
    shutil.rmtree(test_dir)
    print("Тестовая директория удалена")
    
    print("Тест utils завершен успешно!")


if __name__ == "__main__":
    test_utils()