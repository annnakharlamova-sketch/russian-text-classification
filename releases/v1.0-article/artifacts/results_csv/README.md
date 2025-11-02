# Результаты экспериментов

## Файлы в этой папке:

### Основные файлы:
- `evaluation_results.csv` - Полные результаты оценки всех моделей
- `results_summary.csv` - Сводная таблица результатов
- `cv_detailed_results.csv` - Детальные результаты кросс-валидации
- `model_predictions.csv` - Предсказания моделей для графиков
- `learning_curves.csv` - Данные для кривых обучения

### Таблицы для статьи:
- `table_1_model_comparison.csv` - Таблица 1: Сравнение моделей
- `table_2_preprocessing_impact.csv` - Таблица 2: Влияние предобработки  
- `table_3_confidence_intervals.csv` - Таблица 3: Доверительные интервалы

## Структура данных:

### evaluation_results.csv:
- dataset: название датасета (rusentiment, rureviews, taiga_social)
- model: модель (bow_logreg, tfidf_svm, lstm)
- preprocess: пайплайн предобработки (P0, P1, P2, P3)
- accuracy, f1_macro, precision_macro, recall_macro: метрики качества
- f1_macro_ci_lower, f1_macro_ci_upper: 95% доверительный интервал

### results_summary.csv:
- Группировка по dataset, model, preprocess
- Средние значения и стандартные отклонения метрик

Все эксперименты проведены с random_seed=42 для воспроизводимости.
