# Классификация русскоязычных текстов: сравнение классических и нейросетевых методов

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Репозиторий содержит код для воспроизведения экспериментов из научной статьи "Сравнение эффективности классических и нейросетевых методов классификации коротких русскоязычных текстов".

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

## Структура проекта
russian-text-classification/
├── src/                    # Исходный код
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
├── configs/               # Конфигурации
│   └── experiment_config.yaml
├── data/                  # Корпуса данных
├── results/               # Результаты
└── run_experiments.py     # Главный скрипт

## Данные

Для работы необходимо скачать корпуса:

RuSentiment: https://huggingface.co/datasets/MonoHime/ru_sentiment_dataset

RuReviews: https://github.com/sismetanin/rureviews

Taiga Corpus: https://tatianashavrina.github.io/taiga_site/

## Подготовка данных
Разместите скачанные корпуса в папке data/:

text
data/
├── rusentiment/
├── rureviews/
└── taiga/

## Лицензия

MIT License