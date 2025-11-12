"""
Вспомогательные функции для обеспечения воспроизводимости
"""
import datetime
import os
import yaml
import pandas as pd
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import scipy.stats as stats

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


def create_stratified_cv(n_splits=5, shuffle=True, random_state=None):
    """
    Создание 5-кратного стратифицированного кросс-валидатора
    
    Args:
        n_splits (int): Количество фолдов (по умолчанию 5)
        shuffle (bool): Перемешивать ли данные
        random_state (int): Random seed (если None, используется глобальный)
    
    Returns:
        StratifiedKFold: Кросс-валидатор
    """
    if random_state is None:
        random_state = GLOBAL_SEED
    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    
    print(f" Создан {n_splits}-кратный стратифицированный CV:")
    print(f"   - shuffle: {shuffle}")
    print(f"   - random_state: {random_state}")
    
    return cv


def calculate_bootstrap_ci(scores, n_bootstrap=1000, confidence=0.95, random_state=None):
    """
    Расчет 95% доверительного интервала методом перцентильного бутстрэпа
    
    Args:
        scores (array-like): Массив оценок/метрик
        n_bootstrap (int): Количество бутстрэп-выборок
        confidence (float): Уровень доверия (0.95 для 95% CI)
        random_state (int): Random seed
    
    Returns:
        tuple: (lower_bound, upper_bound, bootstrap_samples)
    """
    if random_state is None:
        random_state = GLOBAL_SEED
    
    np.random.seed(random_state)
    
    bootstrap_samples = []
    n_samples = len(scores)
    
    # Генерация бутстрэп-выборок
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(scores, replace=True, n_samples=n_samples)
        bootstrap_samples.append(np.mean(bootstrap_sample))
    
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Расчет перцентилей
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrap_samples, alpha * 100)
    upper_bound = np.percentile(bootstrap_samples, (1 - alpha) * 100)
    
    print(f" Рассчитан {confidence*100}% доверительный интервал (бутстрэп):")
    print(f"   - Количество итераций: {n_bootstrap}")
    print(f"   - Метод: перцентильный бутстрэп")
    print(f"   - Диапазон: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"   - Среднее: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return lower_bound, upper_bound, bootstrap_samples


def perform_cross_validation(model, X, y, cv=None, scoring_func=None):
    """
    Выполнение стратифицированной кросс-валидации
    
    Args:
        model: Модель с fit/predict методами
        X: Признаки
        y: Целевые переменные
        cv: Кросс-валидатор (если None, создается новый)
        scoring_func: Функция для оценки
    
    Returns:
        dict: Результаты CV
    """
    if cv is None:
        cv = create_stratified_cv()
    
    if scoring_func is None:
        from sklearn.metrics import accuracy_score
        scoring_func = accuracy_score
    
    cv_scores = []
    fold_details = []
    
    print(" Запуск стратифицированной кросс-валидации...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказание и оценка
        y_pred = model.predict(X_val)
        score = scoring_func(y_val, y_pred)
        
        cv_scores.append(score)
        fold_details.append({
            'fold': fold,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'score': score
        })
        
        print(f"   Fold {fold}: score = {score:.4f}, "
              f"train/val = {len(train_idx)}/{len(val_idx)}")
    
    # Расчет доверительного интервала
    lower_ci, upper_ci, bootstrap_samples = calculate_bootstrap_ci(cv_scores)
    
    results = {
        'cv_scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'bootstrap_ci': (lower_ci, upper_ci),
        'bootstrap_samples': bootstrap_samples,
        'fold_details': fold_details,
        'cv_params': {
            'n_splits': cv.n_splits,
            'shuffle': cv.shuffle,
            'random_state': cv.random_state
        }
    }
    
    print(f" Результаты CV:")
    print(f"   Среднее: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"   95% CI: [{results['bootstrap_ci'][0]:.4f}, {results['bootstrap_ci'][1]:.4f}]")
    
    return results


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

def save_experiment_results(results_dict, filename=None, results_dir="results"):
    """
    Сохранение результатов эксперимента в стандартизированный CSV
    
    Args:
        results_dict (dict): Словарь с результатами
        filename (str): Имя файла (если None, генерируется автоматически)
        results_dir (str): Директория для результатов
    
    Returns:
        str: Путь к сохраненному файлу
    """
    ensure_dir(results_dir)
    
    # Стандартные столбцы
    required_columns = [
        'dataset', 'model', 'preprocess', 'fold', 'seed', 
        'accuracy', 'macro_f1', 'precision', 'recall', 'train_time_sec'
    ]
    
    # Проверка наличия обязательных полей
    for col in ['dataset', 'model', 'preprocess', 'seed']:
        if col not in results_dict:
            raise ValueError(f"Обязательное поле отсутствует: {col}")
    
    # Создание DataFrame с правильными столбцами
    result_row = {}
    for col in required_columns:
        result_row[col] = results_dict.get(col, None)
    
    df = pd.DataFrame([result_row])
    
    # Генерация имени файла если не указано
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dict['dataset']}_{results_dict['model']}_{results_dict['preprocess']}_seed{results_dict['seed']}.csv"
    
    filepath = os.path.join(results_dir, filename)
    
    # Сохранение (дозапись если файл существует)
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    print(f"✅ Результаты сохранены: {filepath}")
    return filepath


def load_all_results(results_dir="results"):
    """
    Загрузка всех результатов для генерации таблиц
    
    Args:
        results_dir (str): Директория с результатами
    
    Returns:
        pd.DataFrame: Объединенная таблица всех результатов
    """
    all_results = []
    
    if not os.path.exists(results_dir):
        print(f"⚠️ Директория результатов не найдена: {results_dir}")
        return pd.DataFrame()
    
    for file in os.listdir(results_dir):
        if file.endswith('.csv'):
            filepath = os.path.join(results_dir, file)
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                all_results.append(df)
                print(f"📊 Загружено: {file} ({len(df)} строк)")
            except Exception as e:
                print(f"❌ Ошибка загрузки {file}: {e}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"📈 Всего результатов: {len(combined_df)} строк")
        return combined_df
    else:
        print("📭 Нет результатов для загрузки")
        return pd.DataFrame()


if __name__ == "__main__":
    test_reproducibility()