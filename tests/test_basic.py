"""
Базовые unit-тесты для ключевых компонентов
"""

import sys
import os
import pytest

# Добавляем корень проекта в Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_config, ensure_dir
from src.data_preprocessing import DataPreprocessor

def test_config_loading():
    """Тест загрузки конфигурации"""
    config = load_config('configs/experiment_config.yaml')
    assert config is not None
    assert 'data' in config
    assert 'preprocessing' in config
    assert 'models' in config

def test_preprocessing_pipeline():
    """Тест пайплайна предобработки"""
    config = load_config('configs/experiment_config.yaml')
    preprocessor = DataPreprocessor(config)
    
    test_text = "Отличный товар! Очень доволен 123."
    
    # Тест базовой очистки
    result = preprocessor.apply_pipeline(test_text, 'P0')
    assert '123' not in result  # цифры должны быть удалены
    assert '!' not in result    # пунктуация должна быть удалена
    
    # Тест приведения к нижнему регистру
    assert result.islower()

def test_directory_creation():
    """Тест создания директорий"""
    test_dir = 'test_temp_dir'
    ensure_dir(test_dir)
    assert os.path.exists(test_dir)
    
    # cleanup
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)