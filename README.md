# Классификация русскоязычных текстов: сравнение классических и нейросетевых методов

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Версии и окружение:**
- Python 3.10 (протестировано на 3.8-3.11)
- Все зависимости зафиксированы в `requirements.txt`
- Для воспроизводимости используйте `environment.yml` (Conda)

Репозиторий содержит код для воспроизведения экспериментов из научной статьи "Сравнение эффективности классических и нейросетевых методов классификации коротких русскоязычных текстов".

## Воспроизводимость

Для обеспечения полной воспроизводимости результатов:

- **Random seed**: Фиксирован на значении 42
- **Библиотеки**: Версии зафиксированы в `requirements.txt`
- **Алгоритмы**: Детерминированные режимы для PyTorch/CuDNN

### Фиксация Random Seed

Все эксперименты используют `random_seed = 42` для:
- Инициализации весов моделей
- Разделения данных (train/val/test)
- Бутстрэп оценки доверительных интервалов
- DataLoader в PyTorch

```python
from utils import setup_reproducibility
setup_reproducibility(seed=42)  # Установка глобального seed
```

##  Оборудование и параметры моделей

###  Оборудование для экспериментов

**Конфигурация системы:**
- **CPU**: Intel Core i7-12700K
- **GPU**: NVIDIA RTX 3080 (12GB) / NVIDIA A100 (40GB) для LSTM
- **RAM**: 32GB DDR4
- **OS**: Ubuntu 20.04 LTS / Windows 11

**Библиотеки и версии:**
- PyTorch: 2.2.1 (CUDA 11.8)
- scikit-learn: 1.4.2
- Все зависимости зафиксированы в `requirements.txt`

###  Параметры моделей

Все гиперпараметры зафиксированы в конфигурационных файлах:

#### TF-IDF + SVM (`configs/model_svm.yml`)
```yaml
C: 1.0                          # Параметр регуляризации
ngram_range: [1, 2]             # Униграммы + биграммы
max_features: 20000             # Размер словаря
min_df: 3                       # Минимальная частота термина
BoW + Logistic Regression (configs/model_logreg.yml)
yaml
solver: "lbfgs"                 # Алгоритм оптимизации
max_iter: 1000                  # Максимум итераций
ngram_range: [1, 2]             # Униграммы + биграммы  
max_features: 10000             # Размер словаря
min_df: 5                       # Минимальная частота термина
```
LSTM (configs/model_lstm.yml)
```yaml
embedding_dim: 200              # Размер эмбеддингов
hidden_size: 128                # Размер скрытого слоя
num_layers: 1                   # Количество LSTM слоев
bidirectional: true             # Двунаправленная LSTM
dropout: 0.3                    # Dropout регуляризация
batch_size: 32                  # Размер батча
learning_rate: 0.001            # Скорость обучения
max_length: 128                 # Максимальная длина текста
```
##  Время обучения
Среднее время обучения на одном датасете:

Модель	Время обучения	Устройство
TF-IDF + SVM	2-5 минут	CPU
BoW + LogReg	1-3 минуты	CPU
LSTM	15-30 минут	GPU (RTX 3080)
## Воспроизводимость
Random seed: 42 (глобально фиксирован)

Инициализация эмбеддингов: uniform (-0.1, 0.1)

Pad token: индекс 0

Gradient clipping: 1.0

Детерминированные алгоритмы: включены для PyTorch

## Проверка конфигураций
```bash
# Проверка загрузки конфигов
python -c "
from src.models import load_model_config
for model in ['svm', 'logreg', 'lstm']:
    config = load_model_config(model)
    print(f'{model}: {config[\\\"model\\\"][\\\"name\\\"]}')
"
```
# Проверка параметров из статьи
```bash
python scripts/verify_parameters.py
```
## Быстрый старт

### Установка зависимостей
```bash
git clone https://github.com/annnakharlamova-sketch/russian-text-classification.git
cd russian-text-classification
pip install -r requirements.txt
```

### Запуск экспериментов
```bash
python run_experiments.py --all
```

### Отдельные этапы
```bash
# Только предобработка данных
python run_experiments.py --preprocess

# Только классические модели
python run_experiments.py --classical

# Только нейросетевые модели
python run_experiments.py --neural

# Только оценка
python run_experiments.py --evaluate
```
### Сравнение пайплайнов предобработки:
```bash
# Все пайплайны P0-P3 на RuReviews
python run_experiments.py --dataset rureviews --model bow_logreg --preprocess all --seed 42

# Результаты: results/rureviews_bow_logreg_p0_seed42.csv, ...p1..., ...p2..., ...p3...
```

### Кросс-валидация на Taiga:
```bash
# 5-кратная CV на Taiga с базовой предобработкой
python run_experiments.py --dataset taiga --model tfidf_svm --preprocess p0 --cv 5 --seed 42

# Результат: results/taiga_tfidf_svm_p0_cv5_seed42.csv
```

### Полный эксперимент:
```bash
# Все модели, все пайплайны, все датасеты
python run_experiments.py --all --seed 42

# Результаты: results/all_models_evaluation.csv
```

##  Примеры единичных запусков

### Быстрая проверка одной конфигурации:
```bash
# Классическая модель на RuSentiment с лемматизацией
python run_experiments.py --dataset rusentiment --model tfidf_svm --preprocess p3 --cv 5 --seed 42

# Результат: results/rusentiment_tfidf_svm_p3_seed42.csv
```

## Протокол оценки

### Методология

**Кросс-валидация:**
- 5-кратная стратифицированная кросс-валидация
- Усреднение метрик по всем фолдам
- Random seed = 42 для воспроизводимости

**Доверительные интервалы:**
- 95% доверительные интервалы методом перцентильного бутстрэпа
- 1000 бутстрэп выборок для каждой оценки
- Расчет для Accuracy и F1-macro

**Метрики:**
- Accuracy (точность)
- Precision (точность), macro averaging
- Recall (полнота), macro averaging  
- F1-score, macro averaging

### Формат результатов

Результаты сохраняются в `results/evaluation_results.csv` со следующими полями:

```csv
dataset,model,preprocess,seed,accuracy,accuracy_ci_lower,accuracy_ci_upper,
precision_macro,recall_macro,f1_macro,f1_macro_ci_lower,f1_macro_ci_upper,
samples_count
```

Пример записи:

csv
rusentiment,tfidf_svm,P0,42,0.852,0.845,0.859,0.851,0.852,0.851,0.844,0.858,38000

## Воспроизведение таблиц статьи
```bash
# Запуск полной оценки
python run_experiments.py --evaluate

# Генерация сводных таблиц
python -c "
import pandas as pd
from src.evaluation import Evaluator
evaluator = Evaluator({'evaluation': {'cv_folds': 5, 'bootstrap_samples': 1000, 'confidence_interval': 0.95}})
evaluator.generate_summary_table()
"

# Анализ статистической значимости
python scripts/statistical_tests.py
```

##  Воспроизведение графиков из статьи

###  Генерация всех графиков

```bash
# Установка зависимостей для визуализации
pip install matplotlib seaborn

# Запуск генерации всех графиков
python scripts/make_figures.py
```

## Описание графиков
Скрипт создает следующие графики в папке results/figures/:

Основные графики статьи:
roc_rusentiment.png, roc_rureviews.png, roc_taiga_social.png - ROC curves для всех датасетов (Рис. 1)

confmat_tfidf_svm_rusentiment.png и др. - Матрицы ошибок лучших моделей (Рис. 2)

f1_vs_n_rusentiment.png и др. - Кривые обучения F1 vs train size (Рис. 3)

Дополнительные графики:
preprocessing_impact_*.png - Влияние предобработки на качество

model_comparison_all.png - Сравнение моделей по всем датасетам

## Настройка графиков
Графики генерируются в высоком разрешении (300 DPI) и могут быть настроены через:

```python
# Изменение цветов моделей
generator.model_colors = {
    'bow_logreg': '#1f77b4',
    'tfidf_svm': '#ff7f0e',
    'lstm': '#2ca02c'
}

# Изменение размеров графиков
plt.figure(figsize=(12, 8))  # в коде make_figures.py
```

### Структура файлов графиков
```text
results/figures/
├── roc_rusentiment.png          # ROC curves
├── roc_rureviews.png
├── roc_taiga_social.png
├── confmat_tfidf_svm_rusentiment.png  # Confusion matrices
├── confmat_bow_logreg_rureviews.png
├── f1_vs_n_rusentiment.png      # Learning curves
├── f1_vs_n_rureviews.png
├── f1_vs_n_taiga_social.png
├── preprocessing_impact_rusentiment.png
├── preprocessing_impact_rureviews.png
├── preprocessing_impact_taiga_social.png
└── model_comparison_all.png
```

## Воспроизведение конкретных графиков
```bash
# Только ROC curves
python -c "
from scripts.make_figures import FigureGenerator
generator = FigureGenerator()
generator.plot_roc_curves()
"

# Только матрицы ошибок
python -c "
from scripts.make_figures import FigureGenerator  
generator = FigureGenerator()
generator.plot_confusion_matrices()
"

# Только кривые обучения
python -c "
from scripts.make_figures import FigureGenerator
generator = FigureGenerator()
generator.plot_learning_curves()
"
```

## Примечание
Для построения реальных графиков (не демо) необходимо предварительно запустить эксперименты:

```bash
python run_experiments.py --all
```
Это создаст необходимые файлы с предсказаниями и результатами в папке results/.

```txt
# Дополнительные зависимости для визуализации
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.17.0
```

## Доверительные интервалы
В таблицах статьи представлены:

Точечные оценки: Средние значения метрик
95% ДИ: [нижняя граница - верхняя граница]
Метод: Перцентильный бутстрэп (1000 итераций)

Например: F1 = 0.851 [0.844-0.858] означает:

Точечная оценка F1: 0.851
95% доверительный интервал: от 0.844 до 0.858

## Структура проекта после скачивания
```text
russian-text-classification/
├──  data/ # Исходные данные (скачать отдельно)
│ ├──  rusentiment/
│ │ ├──  train.csv # Пример: "text", "sentiment"
│ │ ├──  test.csv # Пример: "text", "sentiment"
│ │ └──  validation.csv # Пример: "text", "sentiment"
│ ├──  rureviews/
│ │ └──  reviews.csv # Пример: "text", "label"
│ └──  taiga_extracted/
│ └──  social_dataset.csv # Пример: "text", "label"
├──  configs/
│ ├──  experiment_config.yaml # Основные настройки
│ ├──  model_svm.yml # Конфиг TF-IDF + SVM
│ └──  preprocess_p0.yaml # Пайплайн базовой очистки
├──  src/ # Исходный код
│ ├──  data_preprocessing.py # Предобработка P0-P3
│ ├──  models.py # BoW+LogReg, TF-IDF+SVM
│ ├──  neural_models.py # LSTM модели
│ ├──  evaluation.py # Оценка с CI
│ └──  utils.py # Вспомогательные функции
├──  scripts/
│ ├──  make_tables.py # Генерация таблиц для статьи
│ └──  make_figures.py # Генерация графиков
├──  results/ # Автогенерируемые результаты
│ ├──  all_models_evaluation.csv # Сводная таблица
│ ├──  figures/ # Рисунки 1-3 для статьи
│ └──  tables/ # Таблицы 1-3 для статьи
├──  run_experiments.py # Главный скрипт
├──  requirements.txt # Зависимости Python
└──  README.md # Эта документация
```

## Данные

Для работы необходимо скачать корпуса:

RuSentiment: https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset

RuReviews: https://github.com/sismetanin/rureviews

Taiga Corpus: https://tatianashavrina.github.io/taiga_site/

## Лицензии и этика

### Датсеты

#### RuSentiment
- **Источник**: [MonoHime/ru_sentiment_dataset](https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset)
- **Лицензия**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Описание**: Датсет для анализа тональности русскоязычных текстов
- **Условия использования**: Требуется указание авторства при использовании

#### RuReviews  
- **Источник**: [sismetanin/rureviews](https://github.com/sismetanin/rureviews)
- **Лицензия**: MIT License
- **Описание**: Отзывы на товары из интернет-магазинов
- **Условия использования**: Свободное использование с указанием источника

#### Taiga Corpus (Social Networks)
- **Источник**: [tatianashavrina/taiga_site](https://tatianashavrina.github.io/taiga_site/)
- **Лицензия**: Creative Commons Attribution-ShareAlike 4.0 International
- **Описание**: Подколлекция социальных сетей из корпуса Taiga
- **Условия использования**: Требуется цитирование оригинальной работы:

```bibtex
@inproceedings{shavrina2018taiga,
    title={The Taiga Corpus: Annotation and Evaluation over the Social Media Texts},
    author={Shavrina, Tatiana and Shapovalova, Olga},
    booktitle={Proceedings of the International Conference on Computational Linguistics and Intellectual Technologies},
    year={2018}
    }
```

### Disclaimer

**ВАЖНО**: Исходные датасеты не включены в этот репозиторий. 
- Не коммитьте сырые данные в Git
- Скачайте датасеты самостоятельно по ссылкам выше
- Разместите их в папке `data/` согласно структуре ниже
- Соблюдайте лицензионные условия оригинальных датасетов

### Проверка структуры данных

После скачивания данных убедитесь в правильной структуре:

```bash
# Проверка RuSentiment
python -c "
import pandas as pd
df = pd.read_csv('data/rusentiment/train.csv')
print(f'RuSentiment: {len(df)} строк')
print('Колонки:', df.columns.tolist())
print('Пример:')
print(df[['text', 'sentiment']].head(2))
"

# Проверка RuReviews  
python -c "
import pandas as pd  
df = pd.read_csv('data/rureviews/reviews.csv')
print(f'RuReviews: {len(df)} строк')
print('Колонки:', df.columns.tolist())
print('Пример:')
print(df[['text', 'label']].head(2))
"

# Проверка Taiga Social
python -c "
import pandas as pd
df = pd.read_csv('data/taiga/social_dataset.csv')  
print(f'Taiga Social: {len(df)} строк')
print('Колонки:', df.columns.tolist())
print('Пример:')
print(df[['text', 'label']].head(2))
"
```
Ожидаемый вывод для каждого датасета:

```bash
RuSentiment: ~190,000 строк, колонки: ['text', 'sentiment']
RuReviews: ~90,000 строк, колонки: ['text', 'label']
Taiga Social: ~30,000 строк, колонки: ['text', 'label']
```
## Подготовка данных
Разместите скачанные корпуса в папке data/:
```text
data/
├── rusentiment/
│ ├── train.csv # ~133,000 примеров (70%)
│ ├── validation.csv # ~19,000 примеров (10%)
│ └── test.csv # ~38,000 примеров (20%)
├── rureviews/
│ └── reviews.csv # ~90,000 примеров
└── taiga/
└── social_dataset.csv # ~30,000 примеров (социальные сети)
```
##  Релизы и артефакты

###  Быстрая проверка для рецензентов

Для удобства проверки мы подготовили релиз со всеми артефактами:

**[ Скачать артефакты v1.0-article](releases/v1.0-article/v1.0-article_artifacts.zip)**

Архив содержит:
- `results_csv/` - Данные для таблиц 1-3 из статьи
- `figures/` - Все графики (Рис. 1-3) в высоком разрешении  
- `configs/` - Конфигурационные файлы экспериментов

###  Содержимое артефактов

####  Таблицы (results_csv/)
- `table_1_model_comparison.csv` - Сравнение моделей (Таблица 1)
- `table_2_preprocessing_impact.csv` - Влияние предобработки (Таблица 2)
- `table_3_confidence_intervals.csv` - Доверительные интервалы (Таблица 3)
- `evaluation_results.csv` - Полные результаты экспериментов

####  Графики (figures/)
- `roc_*.png` - ROC-кривые (Рисунок 1)
- `confmat_*.png` - Матрицы ошибок (Рисунок 2)  
- `f1_vs_n_*.png` - Кривые обучения (Рисунок 3)

####  Конфигурации (configs/)
- Все YAML файлы с гиперпараметрами моделей
- Конфиги предобработки P0-P3
- Настройки эксперимента

###  Подготовка своего релиза

```bash
# Автоматическая подготовка артефактов
python scripts/prepare_release.py

# Ручная проверка содержимого
ls -la releases/v1.0-article/

# Создание архива вручную
cd releases/v1.0-article/ && zip -r v1.0-article_artifacts.zip artifacts/
```

##  Воспроизводимость и CI

[![Russian Text Classification CI](https://github.com/annnakharlamova-sketch/russian-text-classification/actions/workflows/python-app.yml/badge.svg)](https://github.com/annnakharlamova-sketch/russian-text-classification/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/annnakharlamova-sketch/russian-text-classification/branch/main/graph/badge.svg)](https://codecov.io/gh/annnakharlamova-sketch/russian-text-classification)

###  Проверено CI

Проект включает автоматическую проверку воспроизводимости через GitHub Actions:

- **Smoke-тесты**: проверка основных компонентов на toy-датасетах
- **Unit-тесты**: тестирование ключевых функций
- **Покрытие кода**: мониторинг качества тестов
- **Автоматический запуск**: при каждом push и pull request

###  Локальный запуск тестов

```bash
# Создание toy-датасетов
python scripts/create_toy_data.py

# Запуск smoke-тестов
python scripts/run_smoke_tests.py

# Запуск unit-тестов с покрытием
pytest tests/ -v --cov=src

# Запуск полного пайплайна на toy-данных
python scripts/run_full_pipeline_test.py
```
##  Воспроизведение результатов статьи

### Таблицы из научной статьи:
```bash
# Автоматическая генерация таблиц 1-3 как в статье
python scripts/make_tables.py

# Результаты:
# - results/tables/table1_model_comparison.csv    # Сравнение моделей
# - results/tables/table2_preprocessing_impact.csv # Влияние предобработки  
# - results/tables/table3_corpus_comparison.csv   # Сравнение корпусов
```
### Графики из научной статьи:
```bash
# Генерация всех графиков для публикации
python scripts/make_figures.py

# Результаты в results/figures/:
# - figure1_roc_curves.png           # ROC-кривые
# - figure2_confusion_matrices.png   # Матрицы ошибок
# - figure3_learning_curves.png      # Кривые обучения
```

### Ручной запуск CI
В GitHub репозитории:

Перейдите в Actions

Выберите "Russian Text Classification CI"

Нажмите "Run workflow"

Это гарантирует, что пайплайн работает на чистом окружении!

## Лицензия

MIT License