"""
Скрипт для навчання та збереження моделі передбачення виживання на Титаніку.
Використовує оптимальну модель (Good Fit) з ноутбука Chapter_3_Ov_Un.ipynb
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

def prepare_data():
    """Завантажує та підготовлює дані для навчання"""
    print("🚢 Завантажуємо датасет Titanic...")
    
    # Завантажуємо датасет
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    print(f"✅ Датасет завантажено! Кількість записів: {len(df)}")
    
    # Вибираємо важливі колонки
    df_clean = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
    
    # Заповнюємо пропущені значення віку медіаною
    df_clean = df_clean.fillna({'Age': df_clean['Age'].median()})
    
    # Перетворюємо стать на числа (Male=1, Female=0)
    le = LabelEncoder()
    df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
    
    # Видаляємо рядки з пропущеними значеннями
    df_clean = df_clean.dropna()
    
    print(f"✅ Дані підготовлено! Залишилось {len(df_clean)} записів")
    print(f"Ознаки для навчання: Pclass, Sex, Age, SibSp, Parch, Fare")
    print(f"Цільова змінна: Survived (0 = загинув, 1 = вижив)\n")
    
    return df_clean, le

def train_model():
    """Навчає оптимальну модель та зберігає її"""
    print("="*80)
    print("🟢 НАВЧАННЯ ОПТИМАЛЬНОЇ МОДЕЛІ (GOOD FIT)")
    print("="*80)
    
    # Підготовка даних
    df_clean, label_encoder = prepare_data()
    
    # Розділяємо на ознаки (X) та цільову змінну (y)
    X_full = df_clean.drop('Survived', axis=1)
    y_full = df_clean['Survived']
    
    # Розділяємо на train та test (70% train, 30% test)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42
    )
    
    print(f"📊 Розмір тренувального набору: {len(X_train_full)} записів")
    print(f"📊 Розмір тестового набору: {len(X_test_full)} записів\n")
    
    # Навчаємо модель оптимальної складності
    print("🔧 Навчаємо модель...")
    model_goodfit = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    model_goodfit.fit(X_train_full, y_train_full)
    
    # Передбачаємо результати
    y_train_pred = model_goodfit.predict(X_train_full)
    y_test_pred = model_goodfit.predict(X_test_full)
    
    # Обчислюємо точність
    train_accuracy = accuracy_score(y_train_full, y_train_pred)
    test_accuracy = accuracy_score(y_test_full, y_test_pred)
    
    print("📈 РЕЗУЛЬТАТИ НАВЧАННЯ:")
    print(f"   Точність на тренувальних даних: {train_accuracy*100:.1f}%")
    print(f"   Точність на тестових даних: {test_accuracy*100:.1f}%")
    print(f"   Різниця: {abs(train_accuracy - test_accuracy)*100:.1f}%")
    print("\n💡 Модель ОПТИМАЛЬНА! Вона добре працює на обох наборах даних.\n")
    
    # Створюємо папку для моделей, якщо її немає
    os.makedirs('titanic_game/models', exist_ok=True)
    
    # Зберігаємо модель
    model_path = 'titanic_game/models/titanic_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_goodfit, f)
    print(f"✅ Модель збережено: {model_path}")
    
    # Зберігаємо LabelEncoder
    encoder_path = 'titanic_game/models/label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ LabelEncoder збережено: {encoder_path}")
    
    # Зберігаємо інформацію про середні значення для заповнення пропусків
    # (може знадобитися для передбачень)
    feature_stats = {
        'age_median': df_clean['Age'].median(),
        'fare_median': df_clean['Fare'].median(),
    }
    stats_path = 'titanic_game/models/feature_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(feature_stats, f)
    print(f"✅ Статистика ознак збережена: {stats_path}")
    
    print("\n" + "="*80)
    print("✅ НАВЧАННЯ ЗАВЕРШЕНО УСПІШНО!")
    print("="*80)
    print("\n💡 Порада: Для навчального режиму запустіть також utils.train_all_models()")
    print("   або запустіть навчальний режим в app.py - він навчить всі моделі автоматично.")
    print("="*80)
    
    return model_goodfit, label_encoder, feature_stats

if __name__ == "__main__":
    train_model()

