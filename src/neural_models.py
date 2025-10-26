"""
Исправленная версия нейросетевых моделей
"""

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
        token_ids = [self.vocab.get(token, 1) for token in tokens]
        
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
    """LSTM модель для классификации текстов"""
    
    def __init__(self, vocab_size, embedding_dim=200, hidden_size=128, num_classes=2, 
                 num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, 
            bidirectional=bidirectional, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, num_classes)
        )
        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        logits = self.classifier(hidden)
        return logits

class FixedNeuralModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.label_encoder = LabelEncoder()
    
    def build_vocab(self, texts):
        """Построение словаря"""
        print("    Построение словаря...")
        
        for text in texts:
            tokens = str(text).split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        print(f"    Размер словаря: {len(self.vocab)} токенов")
        return self.vocab
    
    def train_lstm(self, X_train, y_train):
        """Обучение LSTM модели"""
        print(" Обучение LSTM...")
        
        try:
            # Используем значения по умолчанию если нет конфига
            embedding_dim = 200
            hidden_size = 128
            batch_size = 32
            learning_rate = 0.001
            epochs = 5
            
            # Пробуем получить из конфига
            try:
                if 'neural' in self.config and 'lstm' in self.config['neural']:
                    lstm_config = self.config['neural']['lstm']
                    embedding_dim = lstm_config.get('embedding_dim', 200)
                    hidden_size = lstm_config.get('hidden_size', 128)
                    batch_size = lstm_config.get('batch_size', 32)
                    learning_rate = lstm_config.get('learning_rate', 0.001)
                    epochs = lstm_config.get('epochs', 5)
            except:
                pass  # Используем значения по умолчанию
            
            # Подготовка данных
            y_encoded = self.label_encoder.fit_transform(y_train)
            num_classes = len(self.label_encoder.classes_)
            
            self.build_vocab(X_train)
            vocab_size = len(self.vocab)
            
            # Создание датасета
            dataset = TextDataset(X_train, y_encoded, self.vocab)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Модель
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, num_classes).to(device)
            
            # Обучение
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
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
                
                print(f"    Эпоха {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
            
            self.models['lstm'] = model
            print(" LSTM обучена успешно!")
            return True
            
        except Exception as e:
            print(f" Ошибка обучения LSTM: {e}")
            return False
    
    def train_all_models(self, X_train, y_train):
        """Обучение всех моделей"""
        return self.train_lstm(X_train, y_train)

def test_fixed_neural():
    """Тест исправленной модели"""
    from utils import load_config
    
    config = load_config('configs/experiment_config.yaml')
    
    X_train = ["отличный фильм", "ужасное кино", "нормально"]
    y_train = [1, 0, 1]
    
    model = FixedNeuralModel(config)
    success = model.train_all_models(X_train, y_train)
    print(" Тест завершен!" if success else " Тест не пройден")

if __name__ == "__main__":
    test_fixed_neural()