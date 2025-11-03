"""
Тесты для моделей машинного обучения
"""

import sys
import os
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_config
from src.models import ClassicalModel
from src.evaluation import Evaluator

def test_classical_model_initialization():
    """Тест инициализации классических моделей"""
    config = load_config('configs/experiment_config.yaml')
    model = ClassicalModel(config)
    
    assert model is not None
    assert hasattr(model, 'models')
    assert hasattr(model, 'vectorizers')
    assert isinstance(model.models, dict)
    assert isinstance(model.vectorizers, dict)

def test_evaluator_initialization():
    """Тест инициализации оценщика"""
    config = load_config('configs/experiment_config.yaml')
    evaluator = Evaluator(config)
    
    assert evaluator is not None
    assert hasattr(evaluator, 'config')
    assert hasattr(evaluator, 'results')

def test_evaluator_metrics():
    """Тест расчета метрик"""
    config = load_config('configs/experiment_config.yaml')
    evaluator = Evaluator(config)
    
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0]  # одна ошибка
    
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'precision_macro' in metrics
    assert 'recall_macro' in metrics
    assert 'f1_macro' in metrics
    assert 0 <= metrics['accuracy'] <= 1