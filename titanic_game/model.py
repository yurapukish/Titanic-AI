"""
Модуль для роботи з навченою моделлю передбачення виживання на Титаніку.
Містить функції для завантаження моделі та зроблення передбачень.
"""

import pickle
import os
import pandas as pd
import numpy as np

# Шлях до папки з моделями
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'titanic_model.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
STATS_PATH = os.path.join(MODELS_DIR, 'feature_stats.pkl')

# Глобальні змінні для кешування моделі
_model = None
_label_encoder = None
_feature_stats = None

def load_model():
    """
    Завантажує навчену модель з файлу.
    Використовує кешування для уникнення повторного завантаження.
    """
    global _model, _label_encoder, _feature_stats
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Модель не знайдено: {MODEL_PATH}\n"
                "Спочатку запустіть train_model.py для навчання моделі."
            )
        
        # Завантажуємо модель
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        
        # Завантажуємо LabelEncoder
        with open(ENCODER_PATH, 'rb') as f:
            _label_encoder = pickle.load(f)
        
        # Завантажуємо статистику ознак
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH, 'rb') as f:
                _feature_stats = pickle.load(f)
        else:
            # Значення за замовчуванням
            _feature_stats = {
                'age_median': 28.0,
                'fare_median': 14.45
            }
    
    return _model, _label_encoder, _feature_stats

def encode_sex(sex: str, label_encoder):
    """
    Перетворює стать з тексту на число.
    
    Args:
        sex: 'Male' або 'Female' або 'Чоловік' або 'Жінка'
        label_encoder: LabelEncoder для перетворення
    
    Returns:
        int: 0 для Female/Жінка, 1 для Male/Чоловік
    """
    # Мапа українських назв на англійські
    sex_map = {
        'чоловік': 'male',
        'жінка': 'female',
        'male': 'male',
        'female': 'female'
    }
    
    sex_lower = sex.lower().strip()
    if sex_lower in sex_map:
        sex_english = sex_map[sex_lower]
    else:
        # Якщо не знайдено, припускаємо male
        sex_english = 'male'
    
    # Перетворюємо через LabelEncoder
    return label_encoder.transform([sex_english])[0]

def prepare_input(pclass, sex, age, sibsp, parch, fare, label_encoder, feature_stats):
    """
    Підготовлює вхідні дані для передбачення.
    
    Args:
        pclass: Клас каюти (1, 2, або 3)
        sex: Стать ('Male'/'Female' або 'Чоловік'/'Жінка')
        age: Вік
        sibsp: Кількість братів/сестер/дружини на борту
        parch: Кількість батьків/дітей на борту
        fare: Вартість квитка
        label_encoder: LabelEncoder для статі
        feature_stats: Статистика ознак для заповнення пропусків
    
    Returns:
        numpy.ndarray: Підготовлений масив для передбачення
    """
    # Перетворюємо стать
    sex_encoded = encode_sex(sex, label_encoder)
    
    # Заповнюємо пропущені значення
    if age is None or np.isnan(age):
        age = feature_stats.get('age_median', 28.0)
    
    if fare is None or np.isnan(fare):
        fare = feature_stats.get('fare_median', 14.45)
    
    # Створюємо масив у правильному порядку: Pclass, Sex, Age, SibSp, Parch, Fare
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    
    return input_data

def predict_survival(pclass, sex, age, sibsp, parch, fare):
    """
    Робить передбачення чи вижив би пасажир.
    
    Args:
        pclass: Клас каюти (1, 2, або 3)
        sex: Стать ('Male'/'Female' або 'Чоловік'/'Жінка')
        age: Вік
        sibsp: Кількість братів/сестер/дружини на борту
        parch: Кількість батьків/дітей на борту
        fare: Вартість квитка
    
    Returns:
        dict: Словник з результатами передбачення:
            - survived: bool - чи вижив
            - probability: float - ймовірність виживання (0-1)
            - prediction_text: str - текстовий опис результату
    """
    # Завантажуємо модель
    model, label_encoder, feature_stats = load_model()
    
    # Підготовлюємо вхідні дані
    input_data = prepare_input(pclass, sex, age, sibsp, parch, fare, label_encoder, feature_stats)
    
    # Робимо передбачення
    prediction = model.predict(input_data)[0]
    
    # Отримуємо ймовірності
    probabilities = model.predict_proba(input_data)[0]
    survival_probability = probabilities[1]  # Ймовірність виживання
    
    # Формуємо результат
    result = {
        'survived': bool(prediction),
        'probability': float(survival_probability),
        'prediction_text': 'Вижив' if prediction == 1 else 'Загинув'
    }
    
    return result

def get_feature_importance():
    """
    Повертає важливість ознак моделі.
    
    Returns:
        dict: Словник з важливістю кожної ознаки
    """
    model, _, _ = load_model()
    
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    importances = model.feature_importances_
    
    feature_importance = dict(zip(feature_names, importances))
    
    # Сортуємо за важливістю
    feature_importance = dict(sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return feature_importance

