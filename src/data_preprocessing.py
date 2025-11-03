"""
Модуль для предобработки текстовых данных
"""

import os
import pandas as pd
import numpy as np
from razdel import tokenize
from pymorphy2 import MorphAnalyzer
from sklearn.model_selection import train_test_split
import re


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.morph = MorphAnalyzer()
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self):
        """Загрузка стоп-слов русского языка"""
        basic_stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
            'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь',
            'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут',
            'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
            'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже',
            'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того',
            'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом',
            'один', 'почти', 'мой', 'тем', 'чтобы', 'неё', 'сейчас', 'были',
            'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
            'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот',
            'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
            'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой',
            'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
        return basic_stop_words
    
    def clean_text(self, text):
        """Очистка текста от HTML-тегов, пунктуации, цифр"""
        if not isinstance(text, str):
            return ""
        # Удаление HTML-тегов
        text = re.sub(r'<[^>]+>', '', text)
        # Удаление цифр
        text = re.sub(r'\d+', '', text)
        # Удаление пунктуации и специальных символов
        text = re.sub(r'[^\w\s]', ' ', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def to_lowercase(self, text):
        """Приведение к нижнему регистру"""
        return text.lower()
    
    def remove_stopwords(self, tokens):
        """Удаление стоп-слов"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Стемминг токенов"""
        # Простой стеммер на основе pymorphy2
        stemmed = []
        for token in tokens:
            parsed = self.morph.parse(token)[0]
            stemmed.append(parsed.normal_form)
        return stemmed
    
    def lemmatize_tokens(self, tokens):
        """Лемматизация токенов"""
        lemmas = []
        for token in tokens:
            parsed = self.morph.parse(token)[0]
            lemmas.append(parsed.normal_form)
        return lemmas
    
    def tokenize_text(self, text):
        """Токенизация текста с помощью razdel"""
        return [token.text for token in tokenize(text)]
    
    def apply_pipeline(self, text, pipeline_name):
        """Применение пайплайна предобработки"""
        pipelines = self.config['preprocessing']['pipelines']
        
        if pipeline_name not in pipelines:
            raise ValueError(f"Неизвестный пайплайн: {pipeline_name}")
        
        pipeline_steps = pipelines[pipeline_name]
        result = str(text)
        
        # Применяем шаги предобработки
        for step in pipeline_steps:
            if step == 'clean_text':
                result = self.clean_text(result)
            elif step == 'lowercase':
                result = self.to_lowercase(result)
        
        # Токенизация и последующие шаги
        tokens = self.tokenize_text(result)
        
        for step in pipeline_steps:
            if step == 'remove_stopwords':
                tokens = self.remove_stopwords(tokens)
            elif step == 'stemming':
                tokens = self.stem_tokens(tokens)
            elif step == 'lemmatization':
                tokens = self.lemmatize_tokens(tokens)
        
        return ' '.join(tokens)
    
    def load_rusentiment(self, data_path):
        """Загрузка корпуса RuSentiment"""
        try:
            # Пробуем разные возможные структуры файлов
            possible_files = ['train.csv', 'test.csv', 'data.csv', 'dataset.csv']
            loaded_data = []
            
            for file in possible_files:
                file_path = os.path.join(data_path, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    print(f"Загружен {file}: {len(df)} примеров")
                    loaded_data.append(df)
            
            if loaded_data:
                combined_df = pd.concat(loaded_data, ignore_index=True)
                print(f"RuSentiment загружен: {len(combined_df)} всего примеров")
                
                # Предполагаем структуру колонок
                if 'text' in combined_df.columns and 'label' in combined_df.columns:
                    return combined_df[['text', 'label']]
                elif 'text' in combined_df.columns and 'sentiment' in combined_df.columns:
                    combined_df = combined_df.rename(columns={'sentiment': 'label'})
                    return combined_df[['text', 'label']]
                else:
                    print("Нестандартная структура RuSentiment, используем первые 2 колонки")
                    return combined_df.iloc[:, :2].rename(columns={combined_df.columns[0]: 'text', combined_df.columns[1]: 'label'})
            else:
                print("Не найдены файлы данных RuSentiment")
                return None
                
        except Exception as e:
            print(f"Ошибка загрузки RuSentiment: {e}")
            return None

    def load_rureviews(self, data_path):
        """Загрузка корпуса RuReviews"""
        try:
            # Пробуем разные возможные структуры файлов
            possible_files = ['reviews.csv', 'data.csv', 'dataset.csv', 'train.csv']
            
            for file in possible_files:
                file_path = os.path.join(data_path, file)
                # ДЕБАГ: выводим путь для отладки
                print(f"    Проверка пути: {file_path}")
                print(f"    Существует: {os.path.exists(file_path)}")
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    print(f"RuReviews загружен из {file}: {len(df)} примеров")
                    print(f"    Колонки: {df.columns.tolist()}")
                    
                    # Предполагаем структуру колонок
                    if 'text' in df.columns and 'label' in df.columns:
                        return df[['text', 'label']]
                    elif 'review' in df.columns and 'rating' in df.columns:
                        df = df.rename(columns={'review': 'text', 'rating': 'label'})
                        return df[['text', 'label']]
                    else:
                        print("Нестандартная структура RuReviews, используем первые 2 колонки")
                        print(f"    Доступные колонки: {df.columns.tolist()}")
                        return df.iloc[:, :2].rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'})
            
            print("Не найдены файлы данных RuReviews")
            return None
                
        except Exception as e:
            print(f"Ошибка загрузки RuReviews: {e}")
            return None

    def load_taiga(self, data_path, max_sentences=None, skip_large_files=False):
        """
        Загрузка корпуса Taiga из UD-форматированных файлов
        
        Args:
            data_path: путь к данным
            max_sentences: ограничение количества предложений (для smoke-тестов)
            skip_large_files: пропускать большие файлы (для smoke-тестов)
        """
        try:
            print(f"Загрузка Taiga из: {data_path}")
            
            if not os.path.exists(data_path):
                print(f"Путь не существует: {data_path}")
                return None
            
            text_files = []
            tagged_path = os.path.join(data_path, 'tagged_texts')
            if os.path.exists(tagged_path):
                for file in os.listdir(tagged_path):
                    if file.endswith('.txt'):
                        text_files.append(os.path.join(tagged_path, file))
            
            for file in os.listdir(data_path):
                if file.endswith('.txt'):
                    text_files.append(os.path.join(data_path, file))
            
            if not text_files:
                print("Не найдены текстовые файлы в Taiga")
                return None
            
            print(f"Найдено файлов: {len(text_files)}")
            
            all_texts = []
            
            for file_path in text_files:
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Размер в ГБ
                    
                    # Для smoke-тестов пропускаем очень большие файлы
                    if skip_large_files and file_size > 0.5:  # >500 МБ
                        print(f"  Пропуск {os.path.basename(file_path)} ({file_size:.1f} ГБ - для smoke-тестов)")
                        continue
                    
                    print(f"  Обработка {os.path.basename(file_path)} ({file_size:.1f} ГБ)...")
                    
                    current_sentence = []
                    line_count = 0
                    file_sentences = 0
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            line_count += 1
                            
                            # Пропускаем мета-строки
                            if line.startswith('#') or not line:
                                if current_sentence:
                                    sentence_text = ' '.join(current_sentence)
                                    if len(sentence_text) > 10:
                                        all_texts.append(sentence_text)
                                        file_sentences += 1
                                        # Останавливаемся если достигли лимита
                                        if max_sentences and len(all_texts) >= max_sentences:
                                            break
                                    current_sentence = []
                                continue
                            
                            # Парсим токены
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                word = parts[1]
                                current_sentence.append(word)
                            
                            # Проверяем лимит
                            if max_sentences and len(all_texts) >= max_sentences:
                                break
                        
                        # Последнее предложение
                        if current_sentence and (not max_sentences or len(all_texts) < max_sentences):
                            sentence_text = ' '.join(current_sentence)
                            if len(sentence_text) > 10:
                                all_texts.append(sentence_text)
                                file_sentences += 1
                    
                    print(f"    {os.path.basename(file_path)}: {file_sentences} предложений")
                    
                    # Если набрали достаточно - выходим
                    if max_sentences and len(all_texts) >= max_sentences:
                        print(f"    Достигнут лимит {max_sentences} предложений")
                        break
                        
                except Exception as e:
                    print(f"  Ошибка чтения {file_path}: {e}")
                    continue  # Продолжаем с другими файлами
            
            if not all_texts:
                print("Не удалось извлечь тексты из Taiga")
                return None
            
            # Создаем DataFrame
            df = pd.DataFrame({
                'text': all_texts,
                'label': [1] * len(all_texts)  # Фиктивные метки
            })
            
            print(f"Taiga загружен: {len(df)} предложений")
            return df
            
        except Exception as e:
            print(f"Ошибка загрузки Taiga: {e}")
            return None

    def load_corpus_data(self, data_path, corpus_name):
        """Загрузка данных корпуса по имени"""
        if corpus_name == 'rusentiment':
            return self.load_rusentiment(data_path)
        elif corpus_name == 'rureviews':
            return self.load_rureviews(data_path)
        elif corpus_name == 'taiga':
            return self.load_taiga(data_path)
        else:
            print(f"Неизвестный корпус: {corpus_name}")
            return None

    def process_corpus(self, corpus_name, data):
        """Обработка конкретного корпуса"""
        print(f"Обработка корпуса: {corpus_name}")
        
        results = {}
        pipelines = self.config['preprocessing']['pipelines'].keys()
        
        for pipeline in pipelines:
            print(f"  Пайплайн: {pipeline}")
            processed_data = data.copy()
            
            # Применяем пайплайн предобработки
            processed_data['processed_text'] = data['text'].apply(
                lambda x: self.apply_pipeline(x, pipeline)
            )
            
            # Удаляем пустые тексты после предобработки
            processed_data = processed_data[processed_data['processed_text'].str.len() > 0]
            
            results[pipeline] = processed_data
        
        return results

    def process_all_corpora(self):
        """Обработка всех корпусов"""
        corpora_config = self.config['data']['corpora']
        all_processed_data = {}
        
        for corpus_name, corpus_config in corpora_config.items():
            data_path = corpus_config['path']
            
            # Загрузка данных
            print(f"\nЗагрузка корпуса: {corpus_name}")
            data = self.load_corpus_data(data_path, corpus_name)
            
            if data is not None and len(data) > 0:
                print(f"Исходные данные: {len(data)} примеров")
                processed_data = self.process_corpus(corpus_name, data)
                all_processed_data[corpus_name] = processed_data
                self.save_processed_data(corpus_name, processed_data)
            else:
                print(f"Корпус {corpus_name} не загружен или пуст")
        
        return all_processed_data

    def save_processed_data(self, corpus_name, processed_data):
        """Сохранение обработанных данных"""
        output_dir = f"processed_data/{corpus_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        for pipeline_name, data in processed_data.items():
            output_path = f"{output_dir}/{pipeline_name}.csv"
            data.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Сохранено: {output_path} ({len(data)} примеров)")
        
        print(f"Данные сохранены в: {output_dir}")

    def prepare_train_val_test(self, data, corpus_config):
        """Подготовка train/validation/test разделов"""
        test_size = corpus_config.get('test_size', 0.2)
        val_size = corpus_config.get('val_size', 0.1)
        random_state = corpus_config.get('random_state', 42)
        
        # Первое разделение: train + temp
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['label'] if 'label' in data.columns else None
        )
        
        # Второе разделение: validation из train
        val_size_adj = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_data,
            test_size=val_size_adj,
            random_state=random_state,
            stratify=train_data['label'] if 'label' in train_data.columns else None
        )
        
        return train_data, val_data, test_data


def test_preprocessor():
    """Тестовая функция для проверки модуля"""
    test_config = {
        'data': {
            'corpora': {
                'rusentiment': {
                    'path': 'data/rusentiment',
                    'text_column': 'text',
                    'label_column': 'sentiment'
                },
                'taiga_social': {
                    'path': 'data/taiga_extracted/home/tsha/social',
                    'text_column': 'text', 
                    'label_column': 'label'
                }
            }
        },
        'preprocessing': {
            'pipelines': {
                'P0': ['clean_text', 'lowercase'],
                'P1': ['clean_text', 'lowercase', 'remove_stopwords']
            }
        }
    }
    
    print("Тестирование модуля предобработки...")
    processor = DataPreprocessor(test_config)
    
    # Тест загрузки rureviews
    print("\n1. Тест загрузки RuReviews:")
    rureviews_data = processor.load_rureviews('data/rureviews')
    if rureviews_data is not None:
        print(f"   RuReviews: Успешно ({len(rureviews_data)} примеров)")
        print(f"   Колонки: {rureviews_data.columns.tolist()}")
        if len(rureviews_data) > 0:
            print(f"   Пример текста: {rureviews_data['text'].iloc[0][:100]}...")
            print(f"   Метка: {rureviews_data['label'].iloc[0]}")
    else:
        print("   RuReviews: Данные не найдены")

    # Тест загрузки rusentiment
    print("\n2. Тест загрузки RuSentiment:")
    rusentiment_data = processor.load_rusentiment('data/rusentiment')
    if rusentiment_data is not None:
        print(f"   RuSentiment: Успешно ({len(rusentiment_data)} примеров)")
    else:
        print("   RuSentiment: Данные не найдены")
    
    # Тест загрузки taiga
    print("\n2. Тест загрузки Taiga:")
    taiga_data = processor.load_taiga('data/taiga_extracted/home/tsha/social')
    if taiga_data is not None:
        print(f"   Taiga: Успешно ({len(taiga_data)} примеров)")
        # Покажем первые 3 примера
        print("   Примеры текстов:")
        for i, text in enumerate(taiga_data['text'].head(3)):
            print(f"     {i+1}. {text[:50]}...")
    else:
        print("   Taiga: Данные не найдены")
    
    # Тест предобработки
    print("\n3. Тест предобработки текста:")
    test_text = "Это пример текста для тестирования 123!"
    processed_p0 = processor.apply_pipeline(test_text, 'P0')
    processed_p1 = processor.apply_pipeline(test_text, 'P1')
    print(f"   Исходный: '{test_text}'")
    print(f"   P0: '{processed_p0}'")
    print(f"   P1: '{processed_p1}'")
    
    print("\nТест модуля предобработки завершен!")


if __name__ == "__main__":
    test_preprocessor()