import sys
sys.path.append('src')

from neural_models import FixedNeuralModel
import yaml
import pandas as pd
import os
import torch
import time

print("=== ЭТАП 3: ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ (LSTM) ===")

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Загружаем конфиг
config = yaml.safe_load(open('configs/experiment_config.yaml', 'r', encoding='utf-8'))

# Создаем папку для моделей
model_dir = "trained_models/lstm"
os.makedirs(model_dir, exist_ok=True)

# Начинаем с одного корпуса для теста
corpus_name = 'rureviews'
pipeline = 'P0'

print(f"\n ОБУЧЕНИЕ LSTM НА: {corpus_name} - {pipeline}")

# Загружаем обработанные данные
data_path = f"processed_data/{corpus_name}/{pipeline}.csv"
df = pd.read_csv(data_path)
print(f"Загружено: {len(df):,} примеров")

# Проверяем классы
unique_labels = df['label'].unique()
num_classes = len(unique_labels)
print(f"Уникальные метки: {unique_labels}")
print(f"Количество классов: {num_classes}")

# Для теста берем подвыборку
sample_size = 5000
if len(df) > sample_size:
    print(f"Берем подвыборку {sample_size} для быстрого теста...")
    df_sample = df.sample(sample_size, random_state=42)
else:
    df_sample = df

X_train = df_sample['text'].tolist()
y_train = df_sample['label'].tolist()

print(f"Данные для обучения: {len(X_train):,} примеров")

try:
    print(" Создание FixedNeuralModel...")
    start_time = time.time()
    
    # Создаем модель
    neural_model = FixedNeuralModel(config)
    
    print("Построение словаря...")
    # Строим словарь
    neural_model.build_vocab(X_train)
    
    print("Начало обучения LSTM...")
    # Обучаем LSTM (только на тренировочных данных)
    success = neural_model.train_lstm(X_train, y_train)
    
    training_time = time.time() - start_time
    
    if success:
        print(f" LSTM обучена за {training_time:.1f} сек")
        
        # Сохраняем модель
        model_path = f"{model_dir}/{corpus_name}_{pipeline}_lstm.pth"
        # Для сохранения модели нужно добавить метод save в класс FixedNeuralModel
        # Пока просто сообщим об успехе
        print(f" Модель обучена успешно (функция сохранения требует доработки)")
        
        # Тестируем на тех же данных для демонстрации
        print(" Тестирование модели на тренировочных данных...")
        # Здесь нужно добавить метод predict или evaluate
        
    else:
        print(f" Ошибка при обучении LSTM")
    
except Exception as e:
    print(f" Ошибка при обучении LSTM: {e}")
    import traceback
    traceback.print_exc()

print(f"\n ТЕСТОВОЕ ОБУЧЕНИЕ LSTM ЗАВЕРШЕНО!")