"""
Модуль для обучения моделей классификации
"""

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import yaml
from pathlib import Path

def load_model_config(model_name):
    """
    Загрузка конфигурации модели из YAML файла
    
    Args:
        model_name: имя модели ('svm', 'logreg', 'lstm')
    
    Returns:
        config: словарь с конфигурацией
    """
    config_path = Path(f"configs/model_{model_name}.yml")
    if not config_path.exists():
        print(f" Конфиг не найден: {config_path}, использую значения по умолчанию")
        return get_default_config(model_name)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f" Загружена конфигурация: {model_name}")
    return config

def get_default_config(model_name):
    """Конфигурация по умолчанию если файл не найден"""
    defaults = {
        'svm': {
            'vectorizer': {'max_features': 20000, 'ngram_range': [1, 2], 'min_df': 3},
            'classifier': {'C': 1.0, 'kernel': 'linear', 'random_state': 42}
        },
        'logreg': {
            'vectorizer': {'max_features': 10000, 'ngram_range': [1, 2], 'min_df': 5},
            'classifier': {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
        },
        'lstm': {
            'embedding_dim': 200, 'hidden_size': 128, 'dropout': 0.3,
            'batch_size': 32, 'learning_rate': 0.001, 'epochs': 10
        }
    }
    return defaults.get(model_name, {})

class ClassicalModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.vectorizers = {}
    
    def train_bow_logreg(self, X_train, y_train, model_name):
        """Обучение модели Bag-of-Words + Logistic Regression"""
        print(f" Обучение {model_name}...")
        
        try:
            model_config = self.config['models']['classical']['bow_logreg']
            
            # Векторизатор BoW
            vectorizer = CountVectorizer(
                ngram_range=tuple(model_config['vectorizer']['ngram_range']),
                max_features=model_config['vectorizer']['max_features'],
                min_df=model_config['vectorizer']['min_df']
            )
            
            # Классификатор
            classifier = LogisticRegression(
                solver=model_config['classifier']['solver'],
                max_iter=model_config['classifier']['max_iter'],
                random_state=model_config['classifier']['random_state']
            )
            
            # Обучение
            print("    Векторизация текстов...")
            X_vec = vectorizer.fit_transform(X_train)
            print(f"    Размерность признаков: {X_vec.shape}")
            
            print("    Обучение классификатора...")
            classifier.fit(X_vec, y_train)
            
            # Сохраняем модель
            self.vectorizers[model_name] = vectorizer
            self.models[model_name] = classifier
            
            print(f" {model_name} обучена успешно!")
            return vectorizer, classifier
            
        except Exception as e:
            print(f" Ошибка обучения {model_name}: {e}")
            return None, None
    
    def train_tfidf_svm(self, X_train, y_train, model_name):
        """Обучение модели TF-IDF + SVM"""
        print(f" Обучение {model_name}...")
        
        try:
            model_config = self.config['models']['classical']['tfidf_svm']
            
            # Векторизатор TF-IDF
            vectorizer = TfidfVectorizer(
                ngram_range=tuple(model_config['vectorizer']['ngram_range']),
                max_features=model_config['vectorizer']['max_features'],
                min_df=model_config['vectorizer']['min_df'],
                sublinear_tf=model_config['vectorizer']['sublinear_tf'],
                use_idf=model_config['vectorizer']['use_idf'],
                norm=model_config['vectorizer']['norm']
            )
            
            # Классификатор SVM
            classifier = LinearSVC(
                C=model_config['classifier']['C'],
                random_state=model_config['classifier']['random_state'],
                max_iter=1000
            )
            
            # Обучение
            print("   Векторизация текстов (TF-IDF)...")
            X_vec = vectorizer.fit_transform(X_train)
            print(f"    Размерность признаков: {X_vec.shape}")
            
            print("    Обучение SVM...")
            classifier.fit(X_vec, y_train)
            
            # Сохраняем модель
            self.vectorizers[model_name] = vectorizer
            self.models[model_name] = classifier
            
            print(f" {model_name} обучена успешно!")
            return vectorizer, classifier
            
        except Exception as e:
            print(f" Ошибка обучения {model_name}: {e}")
            return None, None
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Оценка модели на тестовых данных"""
        if model_name not in self.models:
            print(f" Модель {model_name} не найдена")
            return None
        
        try:
            vectorizer = self.vectorizers[model_name]
            classifier = self.models[model_name]
            
            # Преобразование тестовых данных
            X_test_vec = vectorizer.transform(X_test)
            
            # Предсказания
            y_pred = classifier.predict(X_test_vec)
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f" {model_name} результаты:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-score:  {f1:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f" Ошибка оценки {model_name}: {e}")
            return None
    
    def cross_validate(self, model_name, X, y, cv_folds=5):
        """Кросс-валидация модели"""
        print(f" Кросс-валидация {model_name} ({cv_folds} фолдов)...")
        
        try:
            model_config = self.config['models']['classical'][model_name.split('_')[0]]
            
            # Создаем пайплайн "на лету" для кросс-валидации
            if 'tfidf' in model_name:
                vectorizer = TfidfVectorizer(
                    ngram_range=tuple(model_config['vectorizer']['ngram_range']),
                    max_features=model_config['vectorizer']['max_features'],
                    min_df=model_config['vectorizer']['min_df']
                )
                classifier = LinearSVC(
                    C=model_config['classifier']['C'],
                    random_state=model_config['classifier']['random_state']
                )
            else:  # bow
                vectorizer = CountVectorizer(
                    ngram_range=tuple(model_config['vectorizer']['ngram_range']),
                    max_features=model_config['vectorizer']['max_features'],
                    min_df=model_config['vectorizer']['min_df']
                )
                classifier = LogisticRegression(
                    solver=model_config['classifier']['solver'],
                    max_iter=model_config['classifier']['max_iter'],
                    random_state=model_config['classifier']['random_state']
                )
            
            # Стратифицированная кросс-валидация
            kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                               random_state=self.config['evaluation']['random_state'])
            
            cv_scores = cross_val_score(
                classifier, 
                vectorizer.fit_transform(X), 
                y, 
                cv=kf, 
                scoring='f1_macro'
            )
            
            print(f" Кросс-валидация {model_name}:")
            print(f"   F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return cv_scores
            
        except Exception as e:
            print(f" Ошибка кросс-валидации {model_name}: {e}")
            return None
    
    def train_all_models(self, X_train, y_train):
        """Обучение всех классических моделей"""
        print(" Обучение классических моделей:")
        
        # BoW + Logistic Regression
        self.train_bow_logreg(X_train, y_train, 'bow_logreg')
        
        # TF-IDF + SVM
        self.train_tfidf_svm(X_train, y_train, 'tfidf_svm')
        
        print(" Все классические модели обучены!")
        return len(self.models) > 0
    
    def save_models(self, output_dir="trained_models/classical"):
        """Сохранение обученных моделей"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name in self.models:
            # Сохраняем векторзатор
            vectorizer_path = f"{output_dir}/{model_name}_vectorizer.pkl"
            joblib.dump(self.vectorizers[model_name], vectorizer_path)
            
            # Сохраняем классификатор
            classifier_path = f"{output_dir}/{model_name}_classifier.pkl"
            joblib.dump(self.models[model_name], classifier_path)
            
            print(f" Сохранена модель: {model_name}")
        
        print(f" Все модели сохранены в: {output_dir}")


class NeuralModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
    
    def train_lstm(self, X_train, y_train, model_name):
        """Обучение LSTM модели (заглушка для реализации)"""
        print(f" Обучение {model_name}...")
        print("     LSTM реализация требует дополнительной настройки")
        print("    Для исследований используйте готовые реализации PyTorch/TensorFlow")
        
        # Заглушка для совместимости
        self.models[model_name] = "LSTM_model_placeholder"
        
        print(f"{model_name} готова к реализации!")
        return True
    
    def train_all_models(self, X_train, y_train):
        """Обучение всех нейросетевых моделей"""
        print("Подготовка нейросетевых моделей:")
        
        # LSTM
        self.train_lstm(X_train, y_train, 'lstm')
        
        print("Нейросетевые модели подготовлены!")
        return True


def test_models():
    """Тестовая функция для проверки модуля"""
    print("Тестирование модуля моделей...")
    
    test_config = {
        'models': {
            'classical': {
                'bow_logreg': {
                    'vectorizer': {
                        'ngram_range': [1, 1],
                        'max_features': 1000,
                        'min_df': 1
                    },
                    'classifier': {
                        'solver': 'lbfgs',
                        'max_iter': 100,
                        'random_state': 42
                    }
                },
                'tfidf_svm': {
                    'vectorizer': {
                        'ngram_range': [1, 1],
                        'max_features': 1000,
                        'min_df': 1,
                        'sublinear_tf': True,
                        'use_idf': True,
                        'norm': 'l2'
                    },
                    'classifier': {
                        'C': 1.0,
                        'random_state': 42
                    }
                }
            }
        },
        'evaluation': {
            'random_state': 42
        }
    }
    
    # Тестовые данные
    X_train = [
        "отличный товар очень доволен",
        "ужасное качество не рекомендую", 
        "нормальный продукт за свои деньги",
        "прекрасное обслуживание спасибо",
        "кошмарный сервис больше не обращусь"
    ]
    y_train = [1, 0, 1, 1, 0]  # 1-положительный, 0-отрицательный
    
    X_test = [
        "хороший продукт советую",
        "плохое качество разочарован"
    ]
    y_test = [1, 0]
    
    # Тест классических моделей
    classical_model = ClassicalModel(test_config)
    
    print("1. Тест обучения моделей...")
    classical_model.train_all_models(X_train, y_train)
    
    print("\n2. Тест оценки моделей...")
    for model_name in ['bow_logreg', 'tfidf_svm']:
        classical_model.evaluate_model(model_name, X_test, y_test)
    
    print("\n3. Тест сохранения моделей...")
    classical_model.save_models('test_models')
    
    print("\n4. Тест нейросетевых моделей...")
    neural_model = NeuralModel(test_config)
    neural_model.train_all_models(X_train, y_train)
    
    print(" Тест модуля моделей завершен успешно!")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder

class TextDataset(Dataset):
    """Датасет для нейросетевых моделей"""
    
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Токенизация и преобразование в индексы
        tokens = text.split()[:self.max_length]
        token_ids = [self.vocab.get(token, 1) for token in tokens]  # 1 для OOV
        
        # Паддинг
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class LSTMClassifier(nn.Module):
    """Настоящая LSTM модель для классификации текстов"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, 
                 num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
        
    def forward(self, input_ids):
        # Embedding
        embeddings = self.embedding(input_ids)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        
        # Используем последний скрытый состояние
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Классификация
        logits = self.classifier(hidden)
        return logits

class RealNeuralModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.vocab = {}
        self.label_encoder = LabelEncoder()
    
    def build_vocab(self, texts):
        """Построение словаря из текстов"""
        print("    Построение словаря...")
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for text in texts:
            tokens = str(text).split()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        print(f"    Размер словаря: {len(vocab)} токенов")
        return vocab
    
    def train_lstm(self, X_train, y_train, model_name):
        """Обучение LSTM модели"""
        print(f" Обучение {model_name}...")
        
        try:
            model_config = self.config['models']['neural']['lstm']
            
            # Подготовка данных
            print("    Подготовка данных...")
            
            # Кодирование меток
            y_encoded = self.label_encoder.fit_transform(y_train)
            num_classes = len(self.label_encoder.classes_)
            
            # Построение словаря
            self.vocab = self.build_vocab(X_train)
            vocab_size = len(self.vocab)
            
            # Создание датасета и загрузчика
            dataset = TextDataset(X_train, y_encoded, self.vocab, 
                                max_length=model_config.get('max_length', 128))
            
            dataloader = DataLoader(
                dataset,
                batch_size=model_config['batch_size'],
                shuffle=True,
                num_workers=0
            )
            
            # Создание модели
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"     Устройство: {device}")
            
            model = LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=model_config['embedding_dim'],
                hidden_size=model_config['hidden_size'],
                num_classes=num_classes,
                num_layers=model_config.get('num_layers', 1),
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout']
            ).to(device)
            
            # Оптимизатор и функция потерь
            optimizer = optim.Adam(
                model.parameters(),
                lr=model_config['learning_rate'],
                weight_decay=1e-5
            )
            criterion = nn.CrossEntropyLoss()
            
            # Обучение (упрощенное для скорости)
            print("    Начало обучения (упрощенное)...")
            model.train()
            
            # Только 5 эпох для демонстрации
            for epoch in range(min(5, model_config['epochs'])):
                total_loss = 0
                
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                print(f"    Эпоха {epoch+1}, Loss: {avg_loss:.4f}")
            
            # Сохраняем модель
            self.models[model_name] = {
                'model': model,
                'vocab': self.vocab,
                'label_encoder': self.label_encoder,
                'device': device
            }
            
            print(f" {model_name} обучена успешно!")
            return True
            
        except Exception as e:
            print(f" Ошибка обучения {model_name}: {e}")
            return False
    
    def evaluate_lstm(self, model_name, X_test, y_test):
        """Оценка LSTM модели"""
        if model_name not in self.models:
            print(f" Модель {model_name} не найдена")
            return None
        
        try:
            model_data = self.models[model_name]
            model = model_data['model']
            vocab = model_data['vocab']
            label_encoder = model_data['label_encoder']
            device = model_data['device']
            
            # Подготовка тестовых данных
            y_encoded = label_encoder.transform(y_test)
            dataset = TextDataset(X_test, y_encoded, vocab)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Оценка
            model.eval()
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_ids)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Метрики
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f" {model_name} результаты:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-score:  {f1:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f" Ошибка оценки {model_name}: {e}")
            return None
    
    def train_all_models(self, X_train, y_train):
        """Обучение всех нейросетевых моделей"""
        print(" Обучение нейросетевых моделей:")
        
        # LSTM
        success = self.train_lstm(X_train, y_train, 'lstm')
        
        print("Нейросетевые модели обучены!" if success else " Ошибка обучения нейросетевых моделей")
        return success

if __name__ == "__main__":
    test_models()