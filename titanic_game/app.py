"""
Streamlit додаток для передбачення виживання на Титаніку.
Має два режими: навчальний (про overfitting/underfitting) та ігровий.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import predict_survival, get_feature_importance, load_model
from utils import load_comparison_results

# Налаштування сторінки
st.set_page_config(
    page_title="🚢 Титанік: Навчання та Гра",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🚢 Титанік: Навчання та Гра")
st.markdown("---")

# --- Load dataset ---
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Перемикач режимів
mode = st.sidebar.radio(
    "🎯 Виберіть режим:",
    ["📚 Навчальний режим", "🎮 Ігровий режим"],
    index=0
)

st.sidebar.markdown("---")

# ============================================================================
# НАВЧАЛЬНИЙ РЕЖИМ
# ============================================================================
if mode == "📚 Навчальний режим":
    st.header("📚 Навчальний режим: Overfitting vs Underfitting vs Good Fit")
    
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #1f77b4; margin-top: 0;'>Що ви дізнаєтесь:</h3>
        <ul style='font-size: 16px;'>
            <li>🔴 <strong>Overfitting (Перенавчання)</strong> - що це таке та чому це погано</li>
            <li>🔵 <strong>Underfitting (Недонавчання)</strong> - що це таке та чому це погано</li>
            <li>🟢 <strong>Good Fit (Баланс)</strong> - як знайти оптимальну модель</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Додаємо опис про гру та параметри
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ffc107;'>
        <h3 style='color: #856404; margin-top: 0;'>🎮 Про цю гру:</h3>
        <p style='color: #856404; font-size: 15px;'>
            Ця інтерактивна гра демонструє важливі концепції машинного навчання на прикладі 
            передбачення виживання пасажирів Титаніка. Ми навчили три різні моделі, щоб показати, 
            як різні підходи впливають на якість передбачень.
        </p>
        <h4 style='color: #856404; margin-top: 15px;'>📊 Параметри моделей:</h4>
        <ul style='color: #856404; font-size: 14px;'>
            <li><strong>Ознаки (Features):</strong> Дані про пасажирів - клас каюти, стать, вік, 
                кількість родичів, вартість квитка</li>
            <li><strong>Ціль (Target):</strong> Чи вижив пасажир (так/ні)</li>
            <li><strong>Train Accuracy:</strong> Наскільки точно модель передбачає на даних, 
                на яких вона навчалась</li>
            <li><strong>Test Accuracy:</strong> Наскільки точно модель передбачає на нових даних, 
                яких вона не бачила</li>
            <li><strong>Різниця:</strong> Показує, чи модель добре узагальнює знання 
                (чим менше, тим краще)</li>
        </ul>
        <p style='color: #856404; font-size: 14px; margin-bottom: 0; margin-top: 10px;'>
            <strong>💡 Мета:</strong> Зрозуміти, чому важливо знайти баланс між складністю моделі 
            та її здатністю працювати на нових даних. Після навчання ви зможете спробувати 
            передбачити своє власне виживання в ігровому режимі!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Кнопка для завантаження/навчання моделей
    if st.button("🚀 Почати навчання моделей", type="primary", use_container_width=True):
        with st.spinner("🔧 Навчаємо моделі... Це може зайняти кілька секунд."):
            try:
                results = load_comparison_results()
                st.session_state['comparison_results'] = results
                st.success("✅ Моделі успішно навчені!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Помилка при навчанні моделей: {e}")
    
    # Показуємо результати якщо вони є
    if 'comparison_results' in st.session_state:
        results = st.session_state['comparison_results']
        
        # Вступний текст
        st.markdown("---")
        st.markdown("### 🎓 Що таке Overfitting та Underfitting?")
        
        st.markdown("""
        Коли ми навчаємо модель машинного навчання, ми хочемо, щоб вона добре працювала 
        не тільки на даних, на яких її навчали, але й на нових, небачених даних. 
        Іноді модель може бути занадто складною або занадто простою.
        """)
        
        # Три секції з поясненнями
        tab1, tab2, tab3 = st.tabs(["🔴 Overfitting", "🔵 Underfitting", "🟢 Good Fit"])
        
        # ========== OVERFITTING ==========
        with tab1:
            st.subheader("🔴 Overfitting (Перенавчання)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                #### Що це таке?
                Overfitting виникає, коли модель **занадто складна** або навчена на 
                **занадто малій кількості даних**. Вона "запам'ятовує" тренувальні дані 
                замість того, щоб навчитися загальним закономірностям.
                
                #### Чому це погано?
                - ✅ Модель має високу точність на тренувальних даних
                - ❌ Але погано працює на нових (тестових) даних
                - ⚠️ Велика різниця між train та test accuracy
                
                #### Аналогія:
                Уявіть студента, який вивчив напам'ять всі завдання з підручника, 
                але не розуміє концепцій. На іспиті з новими завданнями він провалиться!
                """)
            
            with col2:
                overfit_data = results['overfitting']
                st.metric("Train Accuracy", f"{overfit_data['train_accuracy']*100:.1f}%")
                st.metric("Test Accuracy", f"{overfit_data['test_accuracy']*100:.1f}%", 
                         delta=f"-{overfit_data['difference']*100:.1f}%", delta_color="inverse")
            
            # Візуалізація
            fig_overfit = go.Figure()
            fig_overfit.add_trace(go.Bar(
                x=['Train', 'Test'],
                y=[overfit_data['train_accuracy']*100, overfit_data['test_accuracy']*100],
                marker_color=['#e74c3c', '#c0392b'],
                text=[f"{overfit_data['train_accuracy']*100:.1f}%", f"{overfit_data['test_accuracy']*100:.1f}%"],
                textposition='outside',
                name='Overfitting'
            ))
            fig_overfit.update_layout(
                title='Overfitting: Велика різниця між Train та Test',
                yaxis_title='Точність (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_overfit, use_container_width=True)
            
            st.info("""
            💡 **Параметри цієї моделі:**
            - Мало даних (тільки 50 прикладів)
            - Дуже глибоке дерево рішень (max_depth=15)
            - Результат: модель "зазубрила" дані, але не може узагальнити
            """)
        
        # ========== UNDERFITTING ==========
        with tab2:
            st.subheader("🔵 Underfitting (Недонавчання)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                #### Що це таке?
                Underfitting виникає, коли модель **занадто проста** або навчена на 
                **поганих ознаках**, які не мають зв'язку з ціллю.
                
                #### Чому це погано?
                - ❌ Модель має низьку точність на тренувальних даних
                - ❌ І також погано працює на тестових даних
                - ⚠️ Модель не може вловити складні закономірності
                
                #### Аналогія:
                Уявіть студента, який навіть не вивчив основи. Він не знає відповіді 
                ні на старі, ні на нові завдання!
                """)
            
            with col2:
                underfit_data = results['underfitting']
                st.metric("Train Accuracy", f"{underfit_data['train_accuracy']*100:.1f}%")
                st.metric("Test Accuracy", f"{underfit_data['test_accuracy']*100:.1f}%")
            
            # Візуалізація
            fig_underfit = go.Figure()
            fig_underfit.add_trace(go.Bar(
                x=['Train', 'Test'],
                y=[underfit_data['train_accuracy']*100, underfit_data['test_accuracy']*100],
                marker_color=['#3498db', '#2980b9'],
                text=[f"{underfit_data['train_accuracy']*100:.1f}%", f"{underfit_data['test_accuracy']*100:.1f}%"],
                textposition='outside',
                name='Underfitting'
            ))
            fig_underfit.update_layout(
                title='Underfitting: Низька точність на обох наборах',
                yaxis_title='Точність (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_underfit, use_container_width=True)
            
            st.info("""
            💡 **Параметри цієї моделі:**
            - Багато даних
            - Але використовується погана ознака (PassengerId - не має зв'язку з виживанням)
            - Результат: модель не може навчитися корисним закономірностям
            """)
        
        # ========== GOOD FIT ==========
        with tab3:
            st.subheader("🟢 Good Fit (Оптимальний баланс)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                #### Що це таке?
                Good Fit - це **оптимальний баланс** між складністю моделі та 
                узагальненням. Модель добре навчена на хороших даних з правильною 
                складністю.
                
                #### Чому це добре?
                - ✅ Модель має високу точність на тренувальних даних
                - ✅ І також добре працює на тестових даних
                - ✅ Мала різниця між train та test accuracy
                
                #### Аналогія:
                Уявіть студента, який вивчив концепції та розуміє матеріал. 
                Він може вирішити і старі, і нові завдання!
                """)
            
            with col2:
                goodfit_data = results['goodfit']
                st.metric("Train Accuracy", f"{goodfit_data['train_accuracy']*100:.1f}%")
                st.metric("Test Accuracy", f"{goodfit_data['test_accuracy']*100:.1f}%")
                st.metric("Різниця", f"{goodfit_data['difference']*100:.1f}%", delta="Мінімальна!")
            
            # Візуалізація
            fig_goodfit = go.Figure()
            fig_goodfit.add_trace(go.Bar(
                x=['Train', 'Test'],
                y=[goodfit_data['train_accuracy']*100, goodfit_data['test_accuracy']*100],
                marker_color=['#2ecc71', '#27ae60'],
                text=[f"{goodfit_data['train_accuracy']*100:.1f}%", f"{goodfit_data['test_accuracy']*100:.1f}%"],
                textposition='outside',
                name='Good Fit'
            ))
            fig_goodfit.update_layout(
                title='Good Fit: Висока точність на обох наборах',
                yaxis_title='Точність (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_goodfit, use_container_width=True)
            
            st.success("""
            💡 **Параметри цієї моделі:**
            - Багато даних (всі доступні записи)
            - Хороші ознаки (Pclass, Sex, Age, SibSp, Parch, Fare)
            - Оптимальна складність (max_depth=5)
            - Результат: модель добре узагальнює та працює на нових даних
            """)
        
        # Порівняння всіх трьох моделей
        st.markdown("---")
        st.subheader("📊 Порівняння всіх трьох моделей")
        
        # Формуємо дані для таблиці з округленням
        comparison_data = {
            'Модель': ['Overfitting', 'Underfitting', 'Good Fit'],
            'Train Accuracy (%)': [
                round(results['overfitting']['train_accuracy']*100, 1),
                round(results['underfitting']['train_accuracy']*100, 1),
                round(results['goodfit']['train_accuracy']*100, 1)
            ],
            'Test Accuracy (%)': [
                round(results['overfitting']['test_accuracy']*100, 1),
                round(results['underfitting']['test_accuracy']*100, 1),
                round(results['goodfit']['test_accuracy']*100, 1)
            ],
            'Різниця (%)': [
                round(results['overfitting']['difference']*100, 1),
                round(results['underfitting']['difference']*100, 1),
                round(results['goodfit']['difference']*100, 1)
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Візуалізація порівняння
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Train Accuracy',
            x=comparison_df['Модель'],
            y=comparison_df['Train Accuracy (%)'],
            marker_color='#3498db',
            text=comparison_df['Train Accuracy (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Test Accuracy',
            x=comparison_df['Модель'],
            y=comparison_df['Test Accuracy (%)'],
            marker_color='#e74c3c',
            text=comparison_df['Test Accuracy (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig_comparison.update_layout(
            title='Порівняння: Overfitting vs Underfitting vs Good Fit',
            xaxis_title='Модель',
            yaxis_title='Точність (%)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Таблиця порівняння з форматуванням
        st.markdown("### 📋 Детальна таблиця порівняння")
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <h4 style='color: #495057; margin-top: 0;'>Пояснення параметрів:</h4>
            <ul style='color: #495057; font-size: 14px;'>
                <li><strong>Train Accuracy</strong> - точність моделі на тренувальних даних (дані, на яких модель навчалась)</li>
                <li><strong>Test Accuracy</strong> - точність моделі на тестових даних (нові дані, яких модель не бачила під час навчання)</li>
                <li><strong>Різниця</strong> - різниця між Train та Test Accuracy (чим менше, тим краще - означає що модель добре узагальнює)</li>
            </ul>
            <p style='color: #495057; font-size: 14px; margin-bottom: 0;'>
                <strong>💡 Ідеальна модель:</strong> висока Test Accuracy + мала Різниця = модель добре працює на нових даних!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Форматуємо таблицю для кращого відображення
        display_df = comparison_df.copy()
        display_df['Train Accuracy (%)'] = display_df['Train Accuracy (%)'].apply(lambda x: f"{x:.1f}%")
        display_df['Test Accuracy (%)'] = display_df['Test Accuracy (%)'].apply(lambda x: f"{x:.1f}%")
        display_df['Різниця (%)'] = display_df['Різниця (%)'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Висновки
        st.markdown("---")
        st.subheader("🎯 Висновки")
        
        st.markdown("""
        <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
            <h4 style='color: #155724; margin-top: 0;'>Ключові моменти:</h4>
            <ul style='color: #155724;'>
                <li><strong>Overfitting</strong>: Велика різниця між train та test accuracy - модель "зазубрила" дані</li>
                <li><strong>Underfitting</strong>: Низька точність на обох наборах - модель занадто проста</li>
                <li><strong>Good Fit</strong>: Висока точність на обох наборах з малою різницею - ідеальний баланс!</li>
            </ul>
            <p style='color: #155724; margin-bottom: 0;'>
                <strong>Мета:</strong> Знайти баланс між складністю моделі та узагальненням, 
                щоб модель добре працювала на нових даних.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("💡 Тепер ви можете перейти до ігрового режиму та спробувати натренувати свою власну модель!")

# ============================================================================
# ІГРОВИЙ РЕЖИМ
# ============================================================================
else:
    st.header("🎮 Інтерактивний режим: Навчи свою модель!")
    
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #1f77b4; margin-top: 0;'>Стань Data Scientist!</h3>
            <p style='font-size: 16px;'>
                Пройди через усі етапи підготовки даних та навчання моделі машинного навчання. 
                На кожному кроці прийми рішення та побач, як воно впливає на точність моделі!
            </p>
            <p style='font-size: 14px; color: #666;'>
                💡 Використовуй підказки (❓), щоб зробити правильний вибір та досягти максимальної точності.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Ініціалізація session_state ---
    if 'game_step' not in st.session_state:
        st.session_state.game_step = 0
    if 'game_choices' not in st.session_state:
        st.session_state.game_choices = {}

    # --- Прогрес-бар ---
    progress = st.session_state.game_step / 6
    st.progress(progress, text=f"Крок {st.session_state.game_step} з 6")
    st.markdown("---")

    # ========== КРОК 0: Початок ==========
    if st.session_state.game_step == 0:
        st.subheader("🎯 Крок 0: Завантаження даних")
        st.markdown("""
        Вітаємо! Ти збираєшся навчити модель машинного навчання для передбачення виживання на Титаніку.
        
        **Що ми маємо:**
        - Дані про пасажирів Титаніка
        - Інформацію про те, хто вижив, а хто ні
        
        **Що ми будемо робити:**
        1. Вибрати важливі ознаки
        2. Очистити дані від помилок
        3. Підготувати дані для навчання
        4. Налаштувати параметри моделі
        5. Навчити модель
        6. Перевірити точність
        """)

        if st.button("🚀 Почати!", type="primary", use_container_width=True):
            st.session_state.game_step = 1
            st.rerun()

    # ========== КРОК 1: Вибір ознак ==========
    elif st.session_state.game_step == 1:
        st.subheader("📊 Крок 1: Вибір важливих ознак")
        st.markdown("""
        Перший крок - вибрати, які дані про пасажирів будуть корисні для передбачення.
        """)

        st.markdown("**Доступні ознаки:**")

        feature_descriptions = {
            'PassengerId': '🆔 Унікальний номер пасажира',
            'Pclass': '🎫 Клас каюти (1-перший/найкращий, 2-другий, 3-третій)',
            'Name': '👤 Ім\'я пасажира',
            'Sex': '⚧️ Стать (male/female)',
            'Age': '🎂 Вік пасажира',
            'SibSp': '👨‍👩‍👧 Кількість братів/сестер/дружини на борту',
            'Parch': '👶 Кількість батьків/дітей на борту',
            'Ticket': '🎟️ Номер квитка',
            'Fare': '💰 Вартість квитка (в фунтах)',
            'Cabin': '🚪 Номер каюти',
            'Embarked': '🌊 Порт посадки (C=Cherbourg, Q=Queenstown, S=Southampton)'
        }

        for feature, description in feature_descriptions.items():
            st.markdown(f"- **{feature}**: {description}")

        st.markdown("---")

        col1, col2 = st.columns([3, 1])

        with col1:
            features = st.multiselect(
                "Вибери ознаки для навчання моделі:",
                options=list(feature_descriptions.keys()),
                default=['Pclass', 'Sex', 'Age'],
                help="Вибери ознаки, які, на твою думку, впливають на виживання"
            )

        with col2:
            show_hint = st.checkbox("❓ Підказка")

        if show_hint:
            st.info("""
            💡 **Підказка:**
            **НЕ корисні:** PassengerId, Name, Ticket, Cabin  
            **Корисні:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked  
            **Оптимальний набір:** Pclass, Sex, Age, SibSp, Parch, Fare
            """)



        # --- Automatically show selected columns ---
        if features:
            st.markdown("### ✅ Твій датасет:")
            cols_to_show = ['Survived'] + features if 'Survived' not in features else features
            st.dataframe(df[cols_to_show].head(1000), use_container_width=True)

        # --- Optional full view button ---
        if st.button("📋 Побачити повну базу даних", use_container_width=True):
            st.markdown("### 📊 Повна база даних:")
            st.dataframe(df.head(1000), use_container_width=True)
            st.info(f"Показано {len(df)} записів")

        # --- Navigation buttons ---
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ Назад", use_container_width=True):
                st.session_state.game_step = 0
                st.rerun()

        with col_btn2:
            if st.button("Далі ➡️", type="primary", use_container_width=True, disabled=len(features) == 0):
                st.session_state.game_choices['features'] = features
                st.session_state.game_choices['cols_to_show'] = cols_to_show
                st.session_state.game_step = 2
                st.rerun()



    # ========== КРОК 2: Обробка пропущених значень віку ==========
    elif st.session_state.game_step == 2:
        st.subheader("🔧 Крок 2: Обробка пропущених значень")

        # ✅ ЗАВЖДИ беремо ОРИГІНАЛЬНІ дані з Кроку 1
        features = st.session_state.game_choices.get('features', [])
        cols_to_show = st.session_state.game_choices.get('cols_to_show', [])

        # Завантажуємо оригінальні дані з CSV (або з session_state якщо вже завантажені)
        if 'original_data' not in st.session_state:
            st.session_state.original_data = pd.read_csv(url)

        # Створюємо НОВУ копію оригінальних даних для цього кроку
        df_step_2 = st.session_state.original_data[cols_to_show].copy()

        st.markdown(f"""Ви обрали ознаки: {features}""")
        age_strategy = "Залишити як є (NaN)"

        if 'Age' in features:
            st.markdown("""
            У даних є пропущені значення віку (деякі пасажири не вказали свій вік).

            **Що робити з пропущеними значеннями?**
            """)

            col1, col2 = st.columns([3, 1])

            with col1:
                age_strategy = st.radio(
                    "Виберіть стратегію:",
                    options=[
                        "Видалити всі рядки з пропущеним віком",
                        "Заповнити медіаною (середнім значенням)",
                        "Заповнити середнім арифметичним",
                        "Залишити як є (NaN)"
                    ],
                    index=1
                )

            with col2:
                show_hint = st.checkbox("❓ Підказка", key="hint_age")

            if show_hint:
                st.info("""
                💡 **Підказка:**
                - **Видалити рядки** - втратимо багато даних (погано!)
                - **Медіана** - найкращий варіант, стійка до викидів ✅
                - **Середнє** - може бути спотворене екстремальними значеннями
                - **Залишити NaN** - модель не зможе навчитися (дуже погано!)

                **Оптимальний вибір:** Заповнити медіаною
                """)

            # ✅ Застосовуємо ОБРАНУ трансформацію до КОПІЇ
            if age_strategy == "Залишити як є (NaN)":
                pass  # нічого не робимо
            elif age_strategy == "Видалити всі рядки з пропущеним віком":
                df_step_2 = df_step_2.dropna(subset=['Age'])
            elif age_strategy == "Заповнити середнім арифметичним":
                df_step_2['Age'] = df_step_2['Age'].fillna(df_step_2['Age'].mean())
            elif age_strategy == "Заповнити медіаною (середнім значенням)":
                df_step_2['Age'] = df_step_2['Age'].fillna(df_step_2['Age'].median())

            st.markdown("### ✅ Твій датасет (після обраної трансформації):")
            st.markdown(f"**Кількість рядків:** {len(df_step_2)}")
            st.dataframe(df_step_2.head(1000), use_container_width=True)

        elif "Age" not in features:
            st.warning("""
            ⚠️ Ви не обрали ознаку **Age**, яка є важливим фактором для прогнозу виживання.

            Ви можете:
            - 📙 **Повернутися назад**, щоб додати `Age`
            - або ➡️ **Продовжити без неї**
            """)

        # ✅ Зберігаємо оброблені дані для наступного кроку
        st.session_state.game_choices['df_step_2'] = df_step_2

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ Назад", use_container_width=True, key="back_2"):
                st.session_state.game_step = 1
                st.rerun()
        with col_btn2:
            if st.button("Далі ➡️", type="primary", use_container_width=True, key="next_2"):
                st.session_state.game_choices['age_strategy'] = age_strategy
                st.session_state.game_step = 3
                st.rerun()

    # ========== КРОК 3: Перетворення категоріальних даних ==========
    elif st.session_state.game_step == 3:
        st.subheader("🔄 Крок 3: Перетворення категоріальних даних")

        # ✅ Беремо ЗБЕРЕЖЕНІ дані з Кроку 2
        df_from_step2 = st.session_state.game_choices.get('df_step_2', None)

        if df_from_step2 is None:
            st.error("❌ Помилка: дані з попереднього кроку не знайдено. Поверніться назад.")
            if st.button("⬅️ Назад", use_container_width=True):
                st.session_state.game_step = 2
                st.rerun()
        else:
            # ✅ ЗАВЖДИ створюємо НОВУ копію
            df_step_3 = df_from_step2.copy()
            original_features = st.session_state.game_choices.get('features', []).copy()
            features = original_features.copy()

            # Визначаємо категоріальні колонки
            categorical_cols = ['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin']
            selected_categorical = [col for col in original_features if
                                    col in categorical_cols and col in df_step_3.columns]

            if not selected_categorical:
                st.info("✅ Ви не обрали категоріальних колонок. Переходимо до наступного кроку.")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("⬅️ Назад", use_container_width=True, key="back_3"):
                        st.session_state.game_step = 2
                        st.rerun()
                with col_btn2:
                    if st.button("Далі ➡️", type="primary", use_container_width=True, key="next_3"):
                        st.session_state.game_choices['df_processed'] = df_step_3
                        st.session_state.game_step = 4
                        st.rerun()
            else:
                st.markdown("""
                Модель машинного навчання працює тільки з числами. 
                Нам потрібно перетворити текстові дані на числа.
                """)

                # ✅ Словник для збереження ПОТОЧНИХ виборів
                current_encodings = {}

                # Обробляємо кожну категоріальну колонку
                for col in selected_categorical:
                    st.markdown(f"---")
                    st.markdown(f"### 📊 Колонка: **{col}**")

                    # Показуємо приклади значень З ОРИГІНАЛЬНИХ даних (до трансформації)
                    original_vals = df_from_step2[col].dropna().unique()[:5]
                    st.markdown(f"**Приклади значень:** {', '.join(map(str, original_vals))}")

                    # SEX
                    if col == 'Sex':
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            sex_encoding = st.radio(
                                "Виберіть метод кодування:",
                                options=[
                                    "Male=1, Female=0",
                                    "Female=1, Male=0",
                                    "За статистикою виживання: Male=1, Female=3",
                                    "Протилежні значення: Male=-1, Female=1",
                                    #"Не використовувати цю колонку"
                                ],
                                index=0,
                                key=f"encoding_sex"
                            )

                        with col2:
                            show_hint = st.checkbox("❓ Підказка", key=f"hint_sex")

                        if show_hint:
                            st.info("""
                            💡 **Підказка:**
                            - Будь-яке числове кодування 0/1 підійде
                            - Стать важлива для прогнозування виживання!
                            **Оптимальний вибір:** Male=1, Female=0 ✅
                            """)

                        current_encodings['Sex'] = sex_encoding

                    # EMBARKED
                    elif col == 'Embarked':
                        st.markdown("Порт посадки: **C** = Cherbourg, **Q** = Queenstown, **S** = Southampton")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            embarked_encoding = st.radio(
                                "Виберіть метод кодування:",
                                options=[
                                    "За алфавітом: C, Q, S → 1, 2, 3",
                                    "За популярністю порту S=3 (найбільше), C=2, Q=1 (найменше) (за кількістю пасажирів)",

                                    #"Не використовувати цю колонку"
                                ],
                                index=0,
                                key=f"encoding_embarked"
                            )

                        with col2:
                            show_hint = st.checkbox("❓ Підказка", key=f"hint_embarked")

                        if show_hint:
                            value_counts = df_from_step2['Embarked'].value_counts()
                            st.info(f"""
                            💡 **Підказка:**
                            - Порт посадки може впливати на клас пасажирів
                            - Розподіл: {value_counts.to_dict()}
                            **Оптимальний вибір:** C=0, Q=1, S=2 ✅
                            """)

                        current_encodings['Embarked'] = embarked_encoding

                    # NAME
                    elif col == 'Name':
                        st.markdown("Приклад: **'Braund, Mr. Owen Harris'**, **'Heikkinen, Miss. Laina'**")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            name_encoding = st.radio(
                                "Виберіть метод обробки:",
                                options=[
                                    "Витягти титулів, сімейного стану (Mr, Mrs, Miss, Master)",
                                    "Підрахувати довжину імені (кількість символів)",
                                    #"Не використовувати цю колонку"
                                ],
                                index=0,
                                key=f"encoding_name"
                            )

                        with col2:
                            show_hint = st.checkbox("❓ Підказка", key=f"hint_name")

                        if show_hint:
                            st.info("""
                            💡 **Підказка:**
                            - **Титул** містить корисну інформацію про стать та статус
                            - **Оптимальний вибір:** Витягти титул ✅
                            """)

                        current_encodings['Name'] = name_encoding

                    # TICKET
                    elif col == 'Ticket':
                        st.markdown("Приклад: **'A/5 21171'**, **'PC 17599'**")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            ticket_encoding = st.radio(
                                "Виберіть метод обробки:",
                                options=[
                                    "Підрахувати довжину квитка",
                                    'Вартість квитка: PC/STON=1 (преміум), A/=2 (середній), Інші=3',
                                    #"Не використовувати цю колонку"
                                ],
                                index=0,
                                key=f"encoding_ticket"
                            )

                        with col2:
                            show_hint = st.checkbox("❓ Підказка", key=f"hint_ticket")

                        if show_hint:
                            st.info("""
                            💡 **Підказка:**
                            - Номер квитка має низьку корисність
                            **Рекомендація:** Не використовувати цю колонку ✅
                            """)

                        current_encodings['Ticket'] = ticket_encoding

                    # CABIN
                    elif col == 'Cabin':
                        cabin_count = df_from_step2['Cabin'].notna().sum()
                        cabin_percent = (cabin_count / len(df_from_step2) * 100)
                        st.markdown(f"Приклад: **'C85'**, **'E46'** | Заповнено: {cabin_count} ({cabin_percent:.1f}%)")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            cabin_encoding = st.radio(
                                "Виберіть метод обробки:",
                                options=[
                                    "Є каюта = 1, Немає каюти = 0",
                                    "Вища палуба = вище число (A/B/C=3, D/E=2, F/G=1, Немає=0)",
                                    "Літера каюти: A=1, B=2, C=3, D=4, E=5, F=6, G=7, Немає=0"
                                    #"Не використовувати цю колонку"
                                ],
                                index=0,
                                key=f"encoding_cabin"
                            )

                        with col2:
                            show_hint = st.checkbox("❓ Підказка", key=f"hint_cabin")

                        if show_hint:
                            st.info(f"""
                            💡 **Підказка:**
                            - Багато пропущених значень ({100 - cabin_percent:.1f}%)
                            - Наявність каюти = вищий клас
                            **Оптимальний вибір:** Є каюта = 1, Немає = 0 ✅
                            """)

                        current_encodings['Cabin'] = cabin_encoding
                # ✅ ЗАСТОСОВУЄМО ВСІ ТРАНСФОРМАЦІЇ ПІСЛЯ збору всіх виборів
                st.markdown("---")
                st.markdown("### 🔄 Застосовуємо трансформації...")

                # Sex - перезаписуємо значення
                if 'Sex' in current_encodings:
                    if current_encodings['Sex'] == "Male=1, Female=0":
                        df_step_3['Sex'] = df_step_3['Sex'].map({'male': 1, 'female': 0})
                    elif current_encodings['Sex'] == "Female=1, Male=0":
                        df_step_3['Sex'] = df_step_3['Sex'].map({'female': 1, 'male': 0})
                    elif current_encodings['Sex'] == "За статистикою виживання: Male=1, Female=3":
                        df_step_3['Sex'] = df_step_3['Sex'].map({'male': 1, 'female': 3})
                    elif current_encodings['Sex'] == "Протилежні значення: Male=-1, Female=1":
                        df_step_3['Sex'] = df_step_3['Sex'].map({'male': -1, 'female': 1})

                # Embarked - перезаписуємо значення
                if 'Embarked' in current_encodings:
                    if current_encodings['Embarked'] == "За алфавітом: C, Q, S → 1, 2, 3":
                        df_step_3['Embarked'] = df_step_3['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
                    elif current_encodings[
                        'Embarked'] == "За популярністю порту S=3 (найбільше), C=2, Q=1 (найменше) (за кількістю пасажирів)":
                        df_step_3['Embarked'] = df_step_3['Embarked'].map({'S': 3, 'C': 2, 'Q': 1})

                # Name - перезаписуємо значення (витягуємо титул або довжину)
                if 'Name' in current_encodings:
                    if current_encodings['Name'] == "Витягти титулів, сімейного стану (Mr, Mrs, Miss, Master)":
                        # Витягуємо титул
                        title_series = df_step_3['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
                        title_mapping = {
                            'Mr': 1,  # Дорослий чоловік
                            'Mrs': 2,  # Одружена жінка
                            'Miss': 3,  # Неодружена жінка/дівчина
                            'Master': 4,  # Хлопчик
                            'Ms': 3,  # Сучасна форма Miss
                            'Mlle': 3,  # Мадемуазель (Miss)
                            'Mme': 2,  # Мадам (Mrs)
                            'Dr': 5,  # Доктор
                            'Rev': 5,  # Преподобний
                            'Col': 5,  # Полковник
                            'Major': 5,  # Майор
                            'Capt': 5,  # Капітан
                            'Sir': 5,  # Сер
                            'Lady': 5,  # Леді
                            'Don': 5,  # Дон
                            'Dona': 5,  # Донья
                            'Countess': 5,  # Графиня
                            'Jonkheer': 5  # Йонкхер (голландський титул)
                        }
                        # ✅ ПЕРЕЗАПИСУЄМО колонку Name числами
                        df_step_3['Name'] = title_series.map(title_mapping).fillna(5)

                    elif current_encodings['Name'] == "Підрахувати довжину імені (кількість символів)":
                        # ✅ ПЕРЕЗАПИСУЄМО колонку Name на довжину
                        df_step_3['Name'] = df_step_3['Name'].str.len()

                # Ticket - перезаписуємо значення
                if 'Ticket' in current_encodings:
                    if current_encodings['Ticket'] == "Підрахувати довжину квитка":
                        # ✅ ПЕРЕЗАПИСУЄМО колонку Ticket на довжину
                        df_step_3['Ticket'] = df_step_3['Ticket'].str.len()

                    elif current_encodings['Ticket'] == 'Вартість квитка: PC/STON=1 (преміум), A/=2 (середній), Інші=3':
                        # Визначаємо тип квитка за префіксом
                        def classify_ticket(ticket):
                            if pd.isna(ticket):
                                return 3  # Немає інформації
                            ticket_str = str(ticket).upper()
                            if 'PC' in ticket_str or 'STON' in ticket_str:
                                return 1  # Преміум
                            elif ticket_str.startswith('A/') or ticket_str.startswith('A.'):
                                return 2  # Середній
                            else:
                                return 3  # Інші


                        # ✅ ПЕРЕЗАПИСУЄМО колонку Ticket категоріями
                        df_step_3['Ticket'] = df_step_3['Ticket'].apply(classify_ticket)

                # Cabin - перезаписуємо значення
                if 'Cabin' in current_encodings:
                    if current_encodings['Cabin'] == "Є каюта = 1, Немає каюти = 0":
                        # ✅ ПЕРЕЗАПИСУЄМО колонку Cabin на 0/1
                        df_step_3['Cabin'] = df_step_3['Cabin'].notna().astype(int)

                    elif current_encodings['Cabin'] == "Вища палуба = вище число (A/B/C=3, D/E=2, F/G=1, Немає=0)":
                        # Визначаємо рівень палуби
                        def classify_deck_level(cabin):
                            if pd.isna(cabin):
                                return 0  # Немає каюти
                            deck = str(cabin)[0].upper()  # Перша літера
                            if deck in ['A', 'B', 'C']:
                                return 3  # Верхні палуби (кращі)
                            elif deck in ['D', 'E']:
                                return 2  # Середні палуби
                            elif deck in ['F', 'G']:
                                return 1  # Нижні палуби
                            else:
                                return 0  # Невідомий формат


                        # ✅ ПЕРЕЗАПИСУЄМО колонку Cabin рівнем палуби
                        df_step_3['Cabin'] = df_step_3['Cabin'].apply(classify_deck_level)

                    elif current_encodings['Cabin'] == "Літера каюти: A=1, B=2, C=3, D=4, E=5, F=6, G=7, Немає=0":
                        # Витягуємо літеру каюти
                        def extract_deck_letter(cabin):
                            if pd.isna(cabin):
                                return 0  # Немає каюти
                            deck = str(cabin)[0].upper()
                            deck_mapping = {
                                'A': 1, 'B': 2, 'C': 3, 'D': 4,
                                'E': 5, 'F': 6, 'G': 7, 'T': 8
                            }
                            return deck_mapping.get(deck, 0)


                        # ✅ ПЕРЕЗАПИСУЄМО колонку Cabin номером літери
                        df_step_3['Cabin'] = df_step_3['Cabin'].apply(extract_deck_letter)

                # Показуємо результат
                st.markdown("---")
                st.markdown("### 📋 Оновлені дані після перетворення")

                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Кількість рядків", len(df_step_3))
                with col_info2:
                    st.metric("Кількість колонок", len(df_step_3.columns))

                st.markdown("**Перші 20 рядків:**")
                st.dataframe(df_step_3.head(20), use_container_width=True)

                # Показуємо які трансформації застосовано
                st.markdown("### ✅ Застосовані трансформації:")
                for col, encoding in current_encodings.items():
                    st.success(f"**{col}**: {encoding}")

                # Перевірка: чи всі колонки числові?
                non_numeric = df_step_3.select_dtypes(exclude=[np.number]).columns.tolist()
                if 'Survived' in non_numeric:
                    non_numeric.remove('Survived')

                if non_numeric:
                    st.warning(f"⚠️ **Увага!** Ще є текстові колонки: {', '.join(non_numeric)}")
                else:
                    st.success("✅ Всі ознаки перетворено на числа! Готово до навчання моделі.")

                # Зберігаємо оброблений DataFrame
                st.session_state.game_choices['df_processed'] = df_step_3

                # Кнопки навігації
                st.markdown("---")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("⬅️ Назад", use_container_width=True, key="back_3"):
                        st.session_state.game_step = 2
                        st.rerun()
                with col_btn2:
                    if st.button("Далі ➡️", type="primary", use_container_width=True, key="next_3"):
                        st.session_state.game_step = 4
                        st.rerun()


    # ========== КРОК 4: Обробка інших пропущених значень ==========
    elif st.session_state.game_step == 4:
        st.subheader("🧹 Крок 4: Фінальне очищення даних")
        
        st.markdown("""
        Можуть залишитися інші пропущені значення в даних.
        
        **Що робити з рядками, які містять пропущені значення?**
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            dropna_strategy = st.radio(
                "Виберіть стратегію:",
                options=[
                    "Видалити всі рядки з будь-якими пропущеними значеннями",
                    "Залишити як є",
                    "Заповнити нулями"
                ],
                index=0
            )
        
        with col2:
            show_hint = st.checkbox("❓ Підказка", key="hint_dropna")
        
        if show_hint:
            st.info("""
            💡 **Підказка:**
            - **Видалити рядки** - найпростіший та надійний спосіб ✅
            - **Залишити** - можуть виникнути помилки при навчанні
            - **Заповнити нулями** - може спотворити дані
            
            **Оптимальний вибір:** Видалити всі рядки з пропущеними значеннями
            """)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ Назад", use_container_width=True, key="back_4"):
                st.session_state.game_step = 3
                st.rerun()
        with col_btn2:
            if st.button("Далі ➡️", type="primary", use_container_width=True, key="next_4"):
                st.session_state.game_choices['dropna_strategy'] = dropna_strategy
                st.session_state.game_step = 5
                st.rerun()
    
    # ========== КРОК 5: Вибір параметрів моделі ==========
    elif st.session_state.game_step == 5:
        st.subheader("⚙️ Крок 5: Налаштування моделі")

        st.markdown("""
        ### Що таке `max_depth`?

        `max_depth` — це **наскільки глибоким може бути дерево рішень**, тобто **скільки запитань поспіль може задати модель**, щоб зробити свій прогноз.

        Пояснення:

        - **Мала глибина (1–2)** — модель задає мало запитань → рішення занадто прості → може часто помилятись.
        - **Дуже велика глибина (15+)** — модель задає занадто багато запитань → починає "зазубрювати" дані → погано працює на нових прикладах.
        - **Середня глибина (3–7)** — модель задає достатньо запитань, але не перебільшує → зазвичай найкращий варіант.

        Обери глибину, яка допоможе моделі робити точні, але не "перенавчені" рішення.
        """)

        col1, col2 = st.columns([3, 1])
        
        with col1:
            max_depth = st.slider(
                "Виберіть max_depth:",
                min_value=1,
                max_value=20,
                value=5,
                help="Максимальна глибина дерева рішень"
            )
            
            # Показуємо попередження залежно від вибору
            if max_depth <= 2:
                st.warning("⚠️ Занадто мала глибина може призвести до underfitting")
            elif max_depth >= 15:
                st.warning("⚠️ Занадто велика глибина може призвести до overfitting")
            else:
                st.success("✅ Хороший вибір для балансу!")
        
        with col2:
            show_hint = st.checkbox("❓ Підказка", key="hint_depth")
        
        if show_hint:
            st.info("""
            💡 **Підказка:**
            Пам'ятаєш навчальний режим?
            - **1-2**: Underfitting (занадто просто)
            - **3-7**: Good Fit (оптимально) ✅
            - **15+**: Overfitting (занадто складно)
            
            **Оптимальний вибір:** 5-7
            """)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⬅️ Назад", use_container_width=True, key="back_5"):
                st.session_state.game_step = 4
                st.rerun()
        with col_btn2:
            if st.button("Далі ➡️", type="primary", use_container_width=True, key="next_5"):
                st.session_state.game_choices['max_depth'] = max_depth
                st.session_state.game_step = 6
                st.rerun()

    # ========== КРОК 6: Навчання та результати ==========
    # ========== КРОК 6: Навчання та результати ==========
    elif st.session_state.game_step == 6:
        st.subheader("🎉 Крок 6: Навчання моделі та результати")

        st.markdown("### 🔧 Твої вибори:")

        choices = st.session_state.game_choices

        # Формуємо детальний список виборів
        choices_data = []

        # 1. Ознаки
        choices_data.append({
            'Крок': '1️⃣ Вибір ознак',
            'Твій вибір': ', '.join(choices.get('features', []))
        })

        # 2. Обробка віку
        age_strategy = choices.get('age_strategy', 'Не обрано')
        choices_data.append({
            'Крок': '2️⃣ Обробка віку',
            'Твій вибір': age_strategy
        })

        # 3. Кодування категоріальних ознак
        encoding_choices = choices.get('encoding_choices', {})
        if encoding_choices:
            for col, encoding in encoding_choices.items():
                choices_data.append({
                    'Крок': f'3️⃣ Кодування: {col}',
                    'Твій вибір': encoding
                })
        else:
            choices_data.append({
                'Крок': '3️⃣ Кодування',
                'Твій вибір': 'Не застосовано'
            })

        # 4. Пропущені значення
        dropna_strategy = choices.get('dropna_strategy', 'Не обрано')
        choices_data.append({
            'Крок': '4️⃣ Пропущені значення',
            'Твій вибір': dropna_strategy
        })

        # 5. Max Depth
        max_depth = choices.get('max_depth', 'Не обрано')
        choices_data.append({
            'Крок': '5️⃣ Max Depth',
            'Твій вибір': str(max_depth)
        })

        choices_df = pd.DataFrame(choices_data)
        st.dataframe(choices_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Кнопка навчання
        if st.button("🚀 Навчити модель!", type="primary", use_container_width=True):
            with st.spinner("🔧 Навчаємо модель на основі твоїх виборів..."):

                # ✅ 1. ОТРИМУЄМО ПІДГОТОВЛЕНІ ДАНІ
                df_processed = st.session_state.game_choices.get('df_processed')

                if df_processed is None:
                    st.error("❌ Дані не підготовлені! Поверніться до попередніх кроків.")
                else:
                    try:
                        # ✅ 2. ВАЛІДАЦІЯ ДАНИХ
                        st.info("🔍 Перевірка даних перед навчанням...")

                        # Перевіряємо наявність Survived
                        if 'Survived' not in df_processed.columns:
                            st.error("❌ Помилка: колонка 'Survived' не знайдена!")
                            st.stop()

                        # Перевіряємо на текстові колонки
                        non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
                        if 'Survived' in non_numeric:
                            non_numeric.remove('Survived')

                        if non_numeric:
                            st.error(f"❌ Помилка: є текстові колонки: {', '.join(non_numeric)}")
                            st.warning("Поверніться до Кроку 3 і перетворіть всі колонки на числа!")
                            st.stop()

                        # Перевіряємо на пропущені значення
                        missing_count = df_processed.isnull().sum().sum()
                        if missing_count > 0:
                            st.warning(f"⚠️ Знайдено {missing_count} пропущених значень. Видаляємо їх...")
                            df_processed = df_processed.dropna()

                        # Перевіряємо чи достатньо даних
                        if len(df_processed) < 50:
                            st.error(f"❌ Занадто мало даних: {len(df_processed)} записів. Потрібно мінімум 50.")
                            st.stop()

                        st.success(f"✅ Дані валідовані! Готово {len(df_processed)} записів для навчання.")

                        # ✅ 3. РОЗДІЛЯЄМО НА X та y
                        X = df_processed.drop('Survived', axis=1)
                        y = df_processed['Survived']

                        st.info(f"📊 Ознаки для навчання: {list(X.columns)}")

                        # ✅ 4. РОЗДІЛЯЄМО НА TRAIN/TEST
                        from sklearn.model_selection import train_test_split

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )

                        st.info(f"🔀 Розділено на Train: {len(X_train)} записів, Test: {len(X_test)} записів")

                        # ✅ 5. НАВЧАЄМО МОДЕЛЬ
                        from sklearn.tree import DecisionTreeClassifier

                        max_depth_val = choices.get('max_depth', 5)

                        model = DecisionTreeClassifier(
                            max_depth=max_depth_val,
                            random_state=42,
                            min_samples_split=5,
                            min_samples_leaf=2
                        )

                        st.info(f"🌳 Навчаємо Decision Tree з max_depth={max_depth_val}...")
                        model.fit(X_train, y_train)

                        # ✅ 6. ОБЧИСЛЮЄМО РЕАЛЬНУ ТОЧНІСТЬ
                        train_accuracy = model.score(X_train, y_train)
                        test_accuracy = model.score(X_test, y_test)

                        # Додаткові метрики
                        from sklearn.metrics import precision_score, recall_score, f1_score

                        y_pred = model.predict(X_test)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)

                        # ✅ 7. ЗБЕРІГАЄМО МОДЕЛЬ
                        st.session_state['trained_model'] = model
                        st.session_state['X_train'] = X_train
                        st.session_state['X_test'] = X_test
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test

                        st.success("✅ Модель успішно навчена!")

                        # ✅ 8. АНАЛІЗУЄМО ВИБОРИ (тільки для feedback, НЕ впливає на оцінку!)
                        feedback = []

                        # Перевірка ознак
                        selected_features = set(choices.get('features', []))
                        optimal_features = {'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'}

                        if 'PassengerId' not in selected_features:
                            feedback.append("✅ Не використовував PassengerId (добре!)")
                        else:
                            feedback.append("⚠️ PassengerId не корисний для прогнозу")

                        useful_selected = len(selected_features.intersection(optimal_features))
                        if useful_selected >= 5:
                            feedback.append(f"✅ Обрав {useful_selected} з 6 найкорисніших ознак")
                        elif useful_selected >= 3:
                            feedback.append(f"⚠️ Обрав {useful_selected} корисних ознак (можна більше)")
                        else:
                            feedback.append(f"❌ Обрав мало корисних ознак: {useful_selected}")

                        # Перевірка обробки віку
                        if 'Age' in selected_features:
                            if 'медіаною' in age_strategy:
                                feedback.append("✅ Використав медіану для віку (оптимально)")
                            elif 'середнім' in age_strategy:
                                feedback.append("⚠️ Середнє працює, але медіана краще")
                            elif 'Видалити' in age_strategy:
                                feedback.append("⚠️ Видалення рядків втрачає багато даних")

                        # Перевірка кодування Sex
                        sex_encoding = encoding_choices.get('Sex', '')
                        if sex_encoding:
                            if 'статистикою' in sex_encoding or 'Female=3' in sex_encoding:
                                feedback.append("✅ Цікавий вибір кодування Sex (враховує статистику)")
                            elif 'Протилежні' in sex_encoding:
                                feedback.append("✅ Креативний вибір кодування Sex")

                        # Перевірка max_depth
                        difference = train_accuracy - test_accuracy

                        if 3 <= max_depth_val <= 7:
                            feedback.append(f"✅ Оптимальний max_depth: {max_depth_val}")
                        elif max_depth_val <= 2:
                            feedback.append(f"⚠️ max_depth={max_depth_val} може бути занадто малим")
                        else:
                            feedback.append(f"⚠️ max_depth={max_depth_val} може призвести до overfitting")

                        # Аналіз РЕАЛЬНИХ результатів моделі
                        if difference > 0.15:
                            feedback.append(
                                f"⚠️ Велика різниця Train-Test ({difference * 100:.1f}%) - ознака overfitting")
                        elif difference < 0.05:
                            feedback.append(f"✅ Мала різниця Train-Test ({difference * 100:.1f}%) - добрий баланс!")

                        if test_accuracy >= 0.80:
                            feedback.append(f"🎉 Відмінна точність на тесті: {test_accuracy * 100:.1f}%!")
                        elif test_accuracy >= 0.75:
                            feedback.append(f"✅ Хороша точність на тесті: {test_accuracy * 100:.1f}%")
                        elif test_accuracy < 0.65:
                            feedback.append(f"⚠️ Низька точність на тесті: {test_accuracy * 100:.1f}%")

                        # ✅ 9. ПОКАЗУЄМО РЕЗУЛЬТАТИ (без змін)
                        st.markdown("---")
                        st.markdown("### 📊 Результати навчання")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Train Accuracy", f"{train_accuracy * 100:.1f}%")
                            st.caption("Це точність моделі на тих даних, на яких вона навчалась.")
                        with col2:
                            st.metric("Test Accuracy", f"{test_accuracy * 100:.1f}%")
                            st.caption("Це точність моделі на нових даних, яких вона ніколи не бачила.")
                        with col3:
                            difference = train_accuracy - test_accuracy
                            delta_color = "inverse" if difference > 0.1 else "normal"
                            st.metric("Різниця", f"{difference * 100:.1f}%",
                                      delta=f"{difference * 100:.1f}%", delta_color=delta_color)
                            st.caption("""
                            ### 🔍 Що означає різниця між Train і Test?
                            - **0–5%** → 🟢 *Чудово!* Модель добре узагальнює і не перенавчена.  
                            - **5–10%** → 🟡 *Нормально.* Є легке перенавчання, але модель працює стабільно.  
                            - **10%+** → 🔴 *Проблема.* Модель перенавчена.
                            """)
                        with col4:
                            st.metric("F1-Score", f"{f1 * 100:.1f}%")
                            st.caption("""
                            ### 🎯 Що таке F1-Score?
                            F1 — це збалансована оцінка точності моделі, яка враховує **і Precision, і Recall**.
                            """)

                        # Детальні метрики
                        with st.expander("📈 Детальні метрики"):
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Precision", f"{precision * 100:.1f}%")
                                st.caption("Наскільки точно модель передбачає *позитивні* приклади.")
                            with metric_col2:
                                st.metric("Recall", f"{recall * 100:.1f}%")
                                st.caption("Яку частку *справжніх позитивів* модель знаходить.")
                            with metric_col3:
                                st.metric("Записів у Train", len(X_train))

                        # ✅ 10. ОЦІНКА НА ОСНОВІ РЕАЛЬНИХ МЕТРИК (ЗМІНЕНО!)
                        st.markdown("---")
                        st.markdown("### 🎯 Оцінка твоєї моделі")

                        # Визначаємо тип fit на основі РЕАЛЬНИХ метрик
                        if difference > 0.15:
                            fit_type = "Overfitting 🔴"
                            fit_explanation = f"Модель занадто добре запам'ятала тренувальні дані (різниця {difference * 100:.1f}%)"
                        elif test_accuracy < 0.70:
                            fit_type = "Underfitting 🔵"
                            fit_explanation = f"Модель занадто проста і не вловлює закономірності (точність {test_accuracy * 100:.1f}%)"
                        else:
                            fit_type = "Good Fit 🟢"
                            fit_explanation = f"Модель добре узагальнює дані! (різниця {difference * 100:.1f}%)"

                        # ОЦІНКА БАЗУЄТЬСЯ ТІЛЬКИ НА РЕАЛЬНИХ МЕТРИКАХ
                        if test_accuracy >= 0.80 and difference < 0.10:
                            st.success(f"""
                            ## 🏆 Відмінно!

                            **Твоя модель: {fit_type}**
                            {fit_explanation}

                            **Результати:**
                            - 🎯 Train Accuracy: {train_accuracy * 100:.1f}%
                            - ✅ Test Accuracy: {test_accuracy * 100:.1f}%
                            - 📊 Різниця: {difference * 100:.1f}%
                            - 🎪 F1-Score: {f1 * 100:.1f}%

                            **Ти справжній Data Scientist!** 🎉
                            """)
                            st.balloons()

                        elif test_accuracy >= 0.75 and difference < 0.15:
                            st.info(f"""
                            ## 👍 Добре!

                            **Твоя модель: {fit_type}**
                            {fit_explanation}

                            **Результати:**
                            - 🎯 Train Accuracy: {train_accuracy * 100:.1f}%
                            - ✅ Test Accuracy: {test_accuracy * 100:.1f}%
                            - 📊 Різниця: {difference * 100:.1f}%
                            - 🎪 F1-Score: {f1 * 100:.1f}%

                            Непогана модель! Є простір для покращення.
                            """)

                        elif test_accuracy >= 0.70:
                            st.warning(f"""
                            ## 🤔 Можна краще!

                            **Твоя модель: {fit_type}**
                            {fit_explanation}

                            **Результати:**
                            - 🎯 Train Accuracy: {train_accuracy * 100:.1f}%
                            - ⚠️ Test Accuracy: {test_accuracy * 100:.1f}%
                            - 📊 Різниця: {difference * 100:.1f}%
                            - 🎪 F1-Score: {f1 * 100:.1f}%

                            Модель працює, але є потенціал для покращення!
                            """)
                        else:
                            st.error(f"""
                            ## 😔 Потрібно покращити

                            **Твоя модель: {fit_type}**
                            {fit_explanation}

                            **Результати:**
                            - 🎯 Train Accuracy: {train_accuracy * 100:.1f}%
                            - ❌ Test Accuracy: {test_accuracy * 100:.1f}%
                            - 📊 Різниця: {difference * 100:.1f}%
                            - 🎪 F1-Score: {f1 * 100:.1f}%

                            Спробуй інші параметри! 💪
                            """)

                        # ✅ 11. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ (без змін)
                        # ... весь код візуалізації залишається

                        # ✅ 12. ДЕТАЛЬНИЙ АНАЛІЗ (без змін)
                        st.markdown("---")
                        st.markdown("### 🔍 Детальний аналіз твоїх виборів")

                        for item in feedback:
                            if '✅' in item:
                                st.success(item)
                            elif '⚠️' in item:
                                st.warning(item)
                            elif '❌' in item:
                                st.error(item)
                            elif '🎉' in item:
                                st.info(item)

                        # Рекомендації БАЗУЮТЬСЯ НА РЕАЛЬНИХ МЕТРИКАХ
                        if test_accuracy < 0.80 or difference > 0.10:
                            st.markdown("---")
                            st.markdown("### 💡 Рекомендації для покращення:")

                            if difference > 0.15:
                                st.info("📌 Overfitting: Зменши max_depth або додай більше даних")
                            if test_accuracy < 0.65:
                                st.info("📌 Underfitting: Збільш max_depth або додай корисні ознаки")
                            if 'PassengerId' in selected_features:
                                st.info("📌 Видали PassengerId - він не допомагає прогнозу")
                            if useful_selected < 4:
                                st.info("📌 Додай більше корисних ознак: Pclass, Sex, Age, SibSp, Parch, Fare")
                    except Exception as e:
                        st.error(f"❌ Помилка: {e}")



            # Кнопки дій
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Спробувати ще раз", use_container_width=True):
                    st.session_state.game_step = 0
                    st.session_state.game_choices = {}
                    st.rerun()
            with col_btn2:
                if st.button("📚 Повернутись до навчання", type="secondary", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()







    # Підвал
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Ця інтерактивна гра допомагає зрозуміти процес створення та оптимізації моделей машинного навчання.</p>
            <p>Навчися робити правильні вибори для досягнення максимальної точності! 🎯</p>
        </div>
        """,
        unsafe_allow_html=True
    )