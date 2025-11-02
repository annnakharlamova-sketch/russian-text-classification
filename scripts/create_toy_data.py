#!/usr/bin/env python3
"""
Создание toy-датасетов для smoke-тестов
"""

import pandas as pd
import os
import numpy as np

def create_toy_rureviews():
    """Создание toy RuReviews датасета"""
    print(" Создание toy RuReviews...")
    
    # Генерируем 1000 примеров сбалансированных отзывов
    np.random.seed(42)
    
    positive_reviews = [
        "отличный товар качество на высоте",
        "прекрасное обслуживание быстро доставили", 
        "хороший продукт соответствует описанию",
        "великолепно рекомендую к покупке",
        "отлично доволен покупкой",
        "качественный товар стоит своих денег",
        "быстрая доставка хорошая упаковка",
        "отличный сервис вежливый персонал",
        "хорошее качество приятно удивлен",
        "рекомендую хороший магазин"
    ]
    
    negative_reviews = [
        "ужасное качество не рекомендую",
        "плохой товар деньги на ветер",
        "кошмарный сервис долгая доставка",
        "некачественный продукт разочарован",
        "плохо не соответствует описанию",
        "бракованный товар вернул обратно", 
        "ужас долго ждал доставку",
        "недоволен плохое обслуживание",
        "качество низкое не советую",
        "проблемы с доставкой не устроил"
    ]
    
    # Создаем сбалансированный датасет
    texts = []
    labels = []
    
    for i in range(500):  # 500 положительных
        base_review = positive_reviews[i % len(positive_reviews)]
        text = f"{base_review} пример {i+1}"
        texts.append(text)
        labels.append(1)
    
    for i in range(500):  # 500 отрицательных  
        base_review = negative_reviews[i % len(negative_reviews)]
        text = f"{base_review} пример {i+1}"
        texts.append(text)
        labels.append(0)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    df.to_csv('data/rureviews/reviews.csv', index=False)
    
    print(f" Toy RuReviews создан: {len(df)} примеров")
    print(f"   Распределение: {df['label'].value_counts().to_dict()}")

def create_toy_rusentiment():
    """Создание toy RuSentiment датасета"""
    print(" Создание toy RuSentiment...")
    
    sentiment_texts = {
        'positive': [
            "люблю этот фильм актеры великолепны",
            "отличная книга рекомендую к прочтению",
            "прекрасная музыка настроение поднимает",
            "замечательный сервис быстро отвечают",
            "восхитительно красивое место"
        ],
        'negative': [
            "ненавижу эту погоду постоянно дождь", 
            "ужасное качество не покупайте",
            "скучный фильм время потрачено зря",
            "плохой сервис не рекомендую",
            "разочарован низким качеством"
        ],
        'neutral': [
            "обычный день ничего особенного",
            "стандартный продукт как везде",
            "нейтральное отношение без эмоций", 
            "обычная погода как всегда",
            "нормально ничего примечательного"
        ]
    }
    
    texts = []
    sentiments = []
    
    for sentiment, examples in sentiment_texts.items():
        for i in range(100):  # по 100 каждого sentiment
            base_text = examples[i % len(examples)]
            text = f"{base_text} пример {i+1}"
            texts.append(text)
            sentiments.append(sentiment)
    
    df = pd.DataFrame({'text': texts, 'sentiment': sentiments})
    df.to_csv('data/rusentiment/train.csv', index=False)
    
    print(f" Toy RuSentiment создан: {len(df)} примеров")
    print(f"   Распределение: {df['sentiment'].value_counts().to_dict()}")

def create_toy_taiga():
    """Создание toy Taiga датасета"""
    print(" Создание toy Taiga...")
    
    social_texts = [
        "сегодня хорошая погода гулял в парке",
        "вчера был на концерте понравилось",
        "работаю над новым проектом интересно",
        "читаю книгу увлекательный сюжет", 
        "смотрю фильм актеры играют хорошо",
        "готовлю ужин пробую новые рецепты",
        "занимаюсь спортом чувствую себя хорошо",
        "встречаюсь с друзьями хорошо пообщались",
        "учу новый язык сложно но интересно",
        "путешествую нравится новые места"
    ]
    
    texts = []
    labels = []
    
    for i in range(1000):
        base_text = social_texts[i % len(social_texts)]
        text = f"{base_text} запись {i+1}"
        texts.append(text)
        labels.append(i % 3)  # 3 класса для разнообразия
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Сохраняем в формате Taiga
    os.makedirs('data/taiga_extracted/social', exist_ok=True)
    df.to_csv('data/taiga_extracted/social/toy_social.csv', index=False)
    
    print(f" Toy Taiga создан: {len(df)} примеров")
    print(f"   Распределение: {df['label'].value_counts().to_dict()}")

def main():
    """Главная функция"""
    print(" СОЗДАНИЕ TOY-ДАТАСЕТОВ ДЛЯ SMOKE-ТЕСТОВ")
    print("=" * 50)
    
    # Создаем директории
    os.makedirs('data/rureviews', exist_ok=True)
    os.makedirs('data/rusentiment', exist_ok=True) 
    os.makedirs('data/taiga_extracted/social', exist_ok=True)
    
    # Создаем toy-датасеты
    create_toy_rureviews()
    create_toy_rusentiment() 
    create_toy_taiga()
    
    print("\n" + "=" * 50)
    print(" ВСЕ TOY-ДАТАСЕТЫ СОЗДАНЫ!")
    print("   Теперь можно запускать smoke-тесты")

if __name__ == "__main__":
    main()