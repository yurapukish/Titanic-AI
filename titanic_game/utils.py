"""
Допоміжні функції для навчального режиму.
Містить функції для навчання та демонстрації overfitting/underfitting/good fit.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def train_all_models():
    """
    Навчає всі три моделі (overfitting, underfitting, good fit) та зберігає їх.
    Повертає результати для візуалізації.
    """
    print("🚢 Завантажуємо датасет Titanic...")
    
    # Завантажуємо датасет
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Підготовка даних для good fit
    df_clean = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
    df_clean = df_clean.fillna({'Age': df_clean['Age'].median()})
    le = LabelEncoder()
    df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
    df_clean = df_clean.dropna()
    
    # ========== OVERFITTING ==========
    print("🔴 Навчаємо модель OVERFITTING...")
    df_small = df_clean.head(50).copy()
    X_small = df_small.drop('Survived', axis=1)
    y_small = df_small['Survived']
    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
        X_small, y_small, test_size=0.4, random_state=42
    )
    
    model_overfit = DecisionTreeClassifier(max_depth=15, min_samples_split=2, random_state=42)
    model_overfit.fit(X_train_small, y_train_small)
    
    train_acc_overfit = accuracy_score(y_train_small, model_overfit.predict(X_train_small))
    test_acc_overfit = accuracy_score(y_test_small, model_overfit.predict(X_test_small))
    
    # ========== UNDERFITTING ==========
    print("🔵 Навчаємо модель UNDERFITTING...")
    df_bad = df[['Survived', 'PassengerId']].copy()
    df_bad = df_bad.dropna()
    X_bad = df_bad[['PassengerId']]
    y_bad = df_bad['Survived']
    X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
        X_bad, y_bad, test_size=0.3, random_state=42
    )
    
    model_underfit = DecisionTreeClassifier(max_depth=3, random_state=42)
    model_underfit.fit(X_train_bad, y_train_bad)
    
    train_acc_underfit = accuracy_score(y_train_bad, model_underfit.predict(X_train_bad))
    test_acc_underfit = accuracy_score(y_test_bad, model_underfit.predict(X_test_bad))
    
    # ========== GOOD FIT ==========
    print("🟢 Навчаємо модель GOOD FIT...")
    X_full = df_clean.drop('Survived', axis=1)
    y_full = df_clean['Survived']
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42
    )
    
    model_goodfit = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    model_goodfit.fit(X_train_full, y_train_full)
    
    train_acc_goodfit = accuracy_score(y_train_full, model_goodfit.predict(X_train_full))
    test_acc_goodfit = accuracy_score(y_test_full, model_goodfit.predict(X_test_full))
    
    # Зберігаємо моделі
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(os.path.join(MODELS_DIR, 'model_overfit.pkl'), 'wb') as f:
        pickle.dump(model_overfit, f)
    
    with open(os.path.join(MODELS_DIR, 'model_underfit.pkl'), 'wb') as f:
        pickle.dump(model_underfit, f)
    
    with open(os.path.join(MODELS_DIR, 'titanic_model.pkl'), 'wb') as f:
        pickle.dump(model_goodfit, f)
    
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    feature_stats = {
        'age_median': df_clean['Age'].median(),
        'fare_median': df_clean['Fare'].median(),
    }
    with open(os.path.join(MODELS_DIR, 'feature_stats.pkl'), 'wb') as f:
        pickle.dump(feature_stats, f)
    
    # Результати для візуалізації
    results = {
        'overfitting': {
            'train_accuracy': train_acc_overfit,
            'test_accuracy': test_acc_overfit,
            'difference': train_acc_overfit - test_acc_overfit,
            'params': 'Мало даних (50)\nГлибоке дерево (depth=15)',
            'color': '#e74c3c'
        },
        'underfitting': {
            'train_accuracy': train_acc_underfit,
            'test_accuracy': test_acc_underfit,
            'difference': abs(train_acc_underfit - test_acc_underfit),
            'params': 'Багато даних\nПогана ознака (PassengerId)',
            'color': '#3498db'
        },
        'goodfit': {
            'train_accuracy': train_acc_goodfit,
            'test_accuracy': test_acc_goodfit,
            'difference': abs(train_acc_goodfit - test_acc_goodfit),
            'params': 'Багато даних\nХороші ознаки + depth=5',
            'color': '#2ecc71'
        }
    }
    
    print("✅ Всі моделі навчені та збережені!")
    
    # Зберігаємо результати для кешування
    save_results(results)
    
    return results

def get_cached_results():
    """
    Спробує завантажити збережені результати з файлу.
    Якщо файлу немає, повертає None.
    """
    results_path = os.path.join(MODELS_DIR, 'comparison_results.pkl')
    if os.path.exists(results_path):
        try:
            with open(results_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_results(results):
    """
    Зберігає результати порівняння у файл.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_path = os.path.join(MODELS_DIR, 'comparison_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

def load_comparison_results(use_cache=True):
    """
    Завантажує результати порівняння моделей.
    Якщо use_cache=True, спробує завантажити з кешу.
    Якщо кешу немає або use_cache=False, навчає моделі.
    """
    if use_cache:
        cached_results = get_cached_results()
        if cached_results is not None:
            return cached_results
    
    # Навчаємо моделі якщо кешу немає
    return train_all_models()

