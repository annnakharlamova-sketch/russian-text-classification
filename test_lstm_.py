"""
Тестирование исправленной LSTM модели
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.neural_models import FixedNeuralModel
from src.utils import load_config

def main():
    print(" Тестирование исправленной LSTM модели...")
    
    config = load_config('configs/experiment_config.yaml')
    
    X_train = [
        "отличный фильм очень понравилось",
        "ужасное кино скучно", 
        "нормальный фильм можно посмотреть",
        "великолепно режиссура",
        "разочарован ожидал большего"
    ]
    y_train = [1, 0, 1, 1, 0]
    
    model = FixedNeuralModel(config)
    success = model.train_all_models(X_train, y_train)
    
    print(" LSTM тест завершен!" if success else " Ошибка LSTM")

if __name__ == "__main__":
    main()