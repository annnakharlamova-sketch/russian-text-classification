"""
Вспомогательные функции для обеспечения воспроизводимости
"""

import os
import yaml
import random
import numpy as np
import torch

# Глобальная переменная для seed
GLOBAL_SEED = 42


def set_global_seed(seed=42):
    """
    Фиксация random seed для полной воспроизводимости экспериментов
    
    Args:
        seed (int): Random seed (по умолчанию 42)
    """
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # если используется multi-GPU
    
    # Установка детерминированных алгоритмов для CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Установка переменных окружения
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f" Random seed установлен глобально: {seed}")


def get_global_seed():
    """Получение текущего global seed"""
    return GLOBAL_SEED


def setup_dataloader_seed(dataloader, seed=None):
    """
    Настройка seed для DataLoader для воспроизводимости
    
    Args:
        dataloader: PyTorch DataLoader
        seed (int): Seed (если None, используется глобальный)
    """
    if seed is None:
        seed = GLOBAL_SEED
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataloader.worker_init_fn = seed_worker
    dataloader.generator = generator
    
    return dataloader


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
    print(f"   Random seed: {GLOBAL_SEED}")
    print(f"   Корпусы: {list(config['data']['corpora'].keys())}")
    print(f"   Пайплайны: {list(config['preprocessing']['pipelines'].keys())}")
    print(f"   Модели: {list(config['models']['classical'].keys())} + LSTM")
    print(f"   Метрики: {config['evaluation']['metrics']}")


def setup_reproducibility(seed=42):
    """
    Полная настройка воспроизводимости (основная функция)
    
    Args:
        seed (int): Random seed
    """
    set_global_seed(seed)
    
    # Дополнительные настройки для NumPy
    np.set_printoptions(precision=8, suppress=True)
    
    # Настройки для PyTorch
    torch.set_printoptions(precision=8)
    
    print(" Воспроизводимость настроена:")
    print(f"   - Python random: {seed}")
    print(f"   - NumPy: {seed}")
    print(f"   - PyTorch: {seed}")
    print(f"   - CuDNN deterministic: True")


def test_reproducibility():
    """Тест воспроизводимости"""
    print(" Тест воспроизводимости...")
    
    setup_reproducibility(42)
    
    # Тест NumPy
    numpy_test = np.random.rand(3)
    print(f"NumPy random: {numpy_test}")
    
    # Тест PyTorch
    torch_test = torch.rand(3)
    print(f"PyTorch random: {torch_test}")
    
    # Тест Python random
    python_test = [random.random() for _ in range(3)]
    print(f"Python random: {python_test}")
    
    print(" Тест воспроизводимости завершен!")


if __name__ == "__main__":
    test_reproducibility()