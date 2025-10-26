"""
Тестирование LSTM модели
"""

import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import RealNeuralModel
from utils import load_config

def main():
    print("🧪 Тестирование LSTM модели...")
    
    # Загрузка конфигурации
    config = load_config('configs/experiment_config.yaml')
    
    # Тестовые данные
    X_train = [
        "отличный фильм очень понравилось актерская игра",
        "ужасное кино скучно и неинтересно", 
        "нормальный фильм можно посмотреть один раз",
        "великолепно режиссура на высшем уровне",
        "разочарован ожидал большего от режиссера"
    ]
    y_train = [1, 0, 1, 1, 0]  # 1-положительный, 0-отрицательный
    
    X_test = [
        "хороший сюжет интересная концовка",
        "плохая актерская игра не впечатлило"
    ]
    y_test = [1, 0]
    
    # Создание и обучение модели
    print("🔄 Создание LSTM модели...")
    model = RealNeuralModel(config)
    
    print("🎯 Обучение LSTM...")
    success = model.train_all_models(X_train, y_train)
    
    if success:
        print("📊 Оценка LSTM...")
        metrics = model.evaluate_lstm('lstm', X_test, y_test)
        
        if metrics:
            print("\n✅ LSTM тест завершен успешно!")
            print("📋 Результаты:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
        else:
            print("❌ Ошибка оценки LSTM")
    else:
        print("❌ Ошибка обучения LSTM")

if __name__ == "__main__":
    main()