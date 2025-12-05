import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Предсказание стоимости автомобиля")

# Инициализируем состояние для хранения истории графиков
if 'eda_feed' not in st.session_state:
    st.session_state['eda_feed'] = []

# Используем cache_resource для загрузки тяжелых объектов (модели), которые не меняются
# Это предотвращает повторную загрузку pickle-файла при каждом клике пользователя
@st.cache_resource
def load_model_pack():
    # Определяем абсолютный путь к файлу, чтобы избежать ошибок при деплое в облако
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'model_pack.pkl')

    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Файл не найден по пути: {file_path}")
        return None

# Используем cache_data для датафреймов
# Streamlit запомнит результат выполнения функции и не будет дергать GitHub лишний раз
@st.cache_data
def load_train_data():
    url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None

data_pack = load_model_pack()
df_train = load_train_data()

if data_pack:
    model = data_pack['model']
    scaler = data_pack['scaler']
    medians = data_pack['medians']
    ohe_columns = data_pack['ohe_columns']
else:
    st.stop()


# Основная функция предобработки для инференса
def preprocess_input(df_input):
    df = df_input.copy()
    
    # Очищаем числовые признаки от единиц измерения (kmpl, CC, bhp)
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.split().str[0], errors='coerce')

    if 'torque' in df.columns and df['torque'].dtype == 'object':
        df['torque'] = pd.to_numeric(df['torque'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    # Заполняем пропуски медианами, которые мы сохранили на этапе обучения
    cols_to_fill = ['mileage', 'engine', 'max_power', 'torque', 'seats']
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    # Убираем признаки, которые не используются в модели
    drop_cols = ['torque', 'name', 'selling_price']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    # Приводим типы данных
    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(int)
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(int).astype(str)

    # OneHotEncoding для категорий
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "unknown"

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # Важный шаг: выравниваем колонки, чтобы их порядок и количество совпадали с трейном
    df = df.reindex(columns=ohe_columns, fill_value=0)
    
    return df

# Функция для очистки данных перед визуализацией
def get_cleaned_data_for_viz(df_raw):
    df = df_raw.copy()
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.split().str[0], errors='coerce')
    
    if 'seats' in df.columns:
        df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
        
    return df


tab1, tab2, tab3 = st.tabs(["📊 Лента EDA", "🤖 Предсказание", "⚖️ Веса модели"])

# Вкладка с EDA
with tab1:
    st.header("Разведочный анализ (Лента событий)")
    st.write("Нажимайте кнопки. Графики строятся на очищенных данных.")

    if df_train is not None:
        df_viz = get_cleaned_data_for_viz(df_train)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
        # Кнопки добавляют тип графика в начало списка в session_state
        with c1:
            if st.button("📈 Гистограмма цены", use_container_width=True):
                st.session_state['eda_feed'].insert(0, "hist_price")
        with c2:
            if st.button("📅 Год vs Цена", use_container_width=True):
                st.session_state['eda_feed'].insert(0, "scatter_year")
        with c3:
            if st.button("🔥 Тепловая карта", use_container_width=True):
                st.session_state['eda_feed'].insert(0, "heatmap")
        with c4:
            if st.button("🧩 Pairplot", use_container_width=True):
                st.session_state['eda_feed'].insert(0, "pairplot")
        with c5:
            if st.button("🗑️ Очистить", use_container_width=True):
                st.session_state['eda_feed'] = []
                st.rerun()

        st.divider()

        # Отрисовка всех графиков из истории
        for graph_type in st.session_state['eda_feed']:
            
            if graph_type == "hist_price":
                st.subheader("Распределение Selling Price")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df_viz['selling_price'], kde=True, ax=ax, color='blue')
                st.pyplot(fig)
                st.markdown("---")

            elif graph_type == "scatter_year":
                st.subheader("Зависимость цены от года выпуска")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.scatterplot(x=df_viz['year'], y=df_viz['selling_price'], ax=ax, color='green')
                st.pyplot(fig)
                st.markdown("---")

            elif graph_type == "heatmap":
                st.subheader("Матрица корреляций (Пирсон)")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_viz.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)
                st.markdown("---")

            elif graph_type == "pairplot":
                st.subheader("Pairplot (Попарные распределения)")
                st.info("Включает основные числовые признаки")
                
                target_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
                fig = sns.pairplot(df_viz[target_cols].dropna())
                st.pyplot(fig)
                st.markdown("---")
    else:
        st.warning("Данные не загружены.")

# Вкладка предсказания
with tab2:
    st.header("Калькулятор стоимости")
    mode = st.radio("Режим ввода:", ["Ввести вручную", "Загрузить CSV"])
    input_df = None

    if mode == "Загрузить CSV":
        uploaded_file = st.file_uploader("CSV файл", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.write("Данные:", input_df.head(3))
    else:
        # Словари для перевода русского интерфейса в значения модели
        fuel_map = {'Дизель': 'Diesel', 'Бензин': 'Petrol', 'Газ (CNG)': 'CNG', 'Газ (LPG)': 'LPG'}
        trans_map = {'Механика': 'Manual', 'Автомат': 'Automatic'}
        seller_map = {'Частник': 'Individual', 'Дилер': 'Dealer', 'Официальный дилер': 'Trustmark Dealer'}
        owner_map = {'Первый': 'First Owner', 'Второй': 'Second Owner', 'Третий': 'Third Owner', 'Четвертый и более': 'Fourth & Above Owner', 'Тест-драйв': 'Test Drive Car'}

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Год выпуска", 1980, 2025, 2017)
            km_driven = st.number_input("Пробег (км)", 0, 5000000, 70000, step=1000)
            
            # Конвертация: л/100км -> kmpl (км на литр)
            # Формула: kmpl = 100 / (л/100км)
            fuel_consump = st.number_input("Расход (л/100 км)", 1.0, 50.0, 8.0, step=0.1)
            mileage_val = 100 / fuel_consump
            mileage_str = f"{mileage_val:.2f} kmpl" # Формируем строку для модели
            
            # Ввод объема в см3, формируем строку "1248 CC"
            engine_vol = st.number_input("Объем двигателя (см³)", 500, 10000, 1600, step=100)
            engine_str = f"{int(engine_vol)} CC"

        with c2:
            # Ввод мощности в л.с., формируем строку "100 bhp" (считаем 1 л.с. ≈ 1 bhp для простоты)
            power_hp = st.number_input("Мощность (л.с.)", 30, 1000, 100, step=5)
            power_str = f"{power_hp} bhp"
            
            seats = st.selectbox("Количество мест", [2,4,5,6,7,8,9,14], index=2)
            
            # Русские селекторы с маппингом
            fuel_ui = st.selectbox("Тип топлива", list(fuel_map.keys()))
            trans_ui = st.selectbox("Коробка передач", list(trans_map.keys()))
            seller_ui = st.selectbox("Продавец", list(seller_map.keys()))
            owner_ui = st.selectbox("Владелец", list(owner_map.keys()))
        
        # Собираем DataFrame, используя английские значения из словарей и сформированные строки
        input_df = pd.DataFrame({
            'name': ['User Car'], 
            'year': [year], 
            'selling_price': [0], 
            'km_driven': [km_driven],
            'fuel': [fuel_map[fuel_ui]],           # Берем значение из словаря
            'seller_type': [seller_map[seller_ui]], # Берем значение из словаря
            'transmission': [trans_map[trans_ui]],  # Берем значение из словаря
            'owner': [owner_map[owner_ui]],         # Берем значение из словаря
            'mileage': [mileage_str],               # Передаем "20.5 kmpl"
            'engine': [engine_str],                 # Передаем "1600 CC"
            'max_power': [power_str],               # Передаем "100 bhp"
            'torque': ['0'], 
            'seats': [seats]
        })

    if st.button("🚀 Рассчитать цену", use_container_width=True) and input_df is not None:
        processed = preprocess_input(input_df)
        X_scaled = scaler.transform(processed)
        preds = model.predict(X_scaled)
        
        if mode == "Ввести вручную":
            st.success(f"Рекомендуемая моделью цена: {preds[0]:,.0f} ₽")
        else:
            input_df['Predicted_Price'] = preds
            st.dataframe(input_df.head())
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV", csv, "predictions.csv", "text/csv")

# Вкладка с весами модели
with tab3:
    st.header("Интерпретация модели")
    coefs = model.coef_
    feats = ohe_columns
    
    # Сортируем признаки по модулю веса
    weights = pd.DataFrame({'Feature': feats, 'Weight': coefs})
    weights['Abs_Weight'] = weights['Weight'].abs()
    weights = weights.sort_values(by='Abs_Weight', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # hue и legend нужны для корректной работы в новых версиях seaborn
    sns.barplot(data=weights, x='Weight', y='Feature', hue='Feature', legend=False, palette='viridis', ax=ax)
    plt.title("Топ-15 признаков")
    st.pyplot(fig)