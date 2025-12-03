import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 0. НАСТРОЙКА СТРАНИЦЫ И СОСТОЯНИЯ
# --------------------------------------------------------
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Предсказание стоимости автомобиля")

if 'eda_feed' not in st.session_state:
    st.session_state['eda_feed'] = []

# --------------------------------------------------------
# 1. ЗАГРУЗКА
# --------------------------------------------------------

@st.cache_resource
def load_model_pack():
    # Получаем абсолютный путь к папке, где лежит app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Собираем полный путь к файлу модели
    file_path = os.path.join(current_dir, 'model_pack.pkl')

    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Файл не найден по пути: {file_path}")
        return None

@st.cache_data
def load_train_data():
    url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Не удалось загрузить данные с GitHub: {e}")
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

# --------------------------------------------------------
# 2. ФУНКЦИИ ПРЕДОБРАБОТКИ
# --------------------------------------------------------

# А) Для предсказания (Точная копия логики ноутбука)
def preprocess_input(df_input):
    df = df_input.copy()
    
    # Очистка строк
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.split().str[0], errors='coerce')

    if 'torque' in df.columns and df['torque'].dtype == 'object':
        df['torque'] = pd.to_numeric(df['torque'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    # Заполнение
    cols_to_fill = ['mileage', 'engine', 'max_power', 'torque', 'seats']
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    # Удаление
    drop_cols = ['torque', 'name', 'selling_price']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    # Типы
    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(int)
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(int).astype(str)

    # OHE
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "unknown"

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    df = df.reindex(columns=ohe_columns, fill_value=0)
    
    return df

# Б) Для визуализации (EDA) - просто чистим числа, не удаляем колонки
def get_cleaned_data_for_viz(df_raw):
    df = df_raw.copy()
    # Превращаем строки в числа
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.split().str[0], errors='coerce')
    
    # seats тоже делаем числом для графиков
    if 'seats' in df.columns:
        df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
        
    return df

# --------------------------------------------------------
# 3. ИНТЕРФЕЙС
# --------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Лента EDA", "🤖 Предсказание", "⚖️ Веса модели"])

# === ВКЛАДКА 1: EDA ===
with tab1:
    st.header("Разведочный анализ (Лента событий)")
    st.write("Нажимайте кнопки. Графики строятся на очищенных данных (как в ноутбуке).")

    if df_train is not None:
        # 1. Сразу готовим чистые данные для графиков
        df_viz = get_cleaned_data_for_viz(df_train)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
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
                # Теперь считаем по df_viz, где engine и power уже числа
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_viz.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)
                st.markdown("---")

            elif graph_type == "pairplot":
                st.subheader("Pairplot (Попарные распределения)")
                st.info("Включает: year, price, km_driven, mileage, engine, max_power, seats")
                
                # Выбираем те же колонки, что и в ДЗ
                target_cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
                
                # dropna нужен, так как pairplot не любит пропуски, которые появились после очистки строк
                fig = sns.pairplot(df_viz[target_cols].dropna())
                st.pyplot(fig)
                st.markdown("---")
    else:
        st.warning("Данные не загружены.")

# === ВКЛАДКА 2: ПРЕДСКАЗАНИЕ ===
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
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Год", 1980, 2025, 2017)
            km_driven = st.number_input("Пробег (км)", 0, 5000000, 70000)
            mileage = st.text_input("Расход (напр. '20 kmpl')", "20 kmpl")
            engine = st.text_input("Двигатель (напр. '1248 CC')", "1248 CC")
        with c2:
            max_power = st.text_input("Мощность (напр. '80 bhp')", "80 bhp")
            seats = st.selectbox("Мест", [2,4,5,6,7,8,9,14], index=2)
            fuel = st.selectbox("Топливо", ['Diesel', 'Petrol', 'CNG', 'LPG'])
            trans = st.selectbox("Коробка", ['Manual', 'Automatic'])
            seller = st.selectbox("Продавец", ['Individual', 'Dealer', 'Trustmark Dealer'])
            owner = st.selectbox("Владелец", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
        
        input_df = pd.DataFrame({
            'name': ['User Car'], 'year': [year], 'selling_price': [0], 'km_driven': [km_driven],
            'fuel': [fuel], 'seller_type': [seller], 'transmission': [trans], 'owner': [owner],
            'mileage': [mileage], 'engine': [engine], 'max_power': [max_power], 'torque': ['0'], 'seats': [seats]
        })

    if st.button("🚀 Рассчитать цену", use_container_width=True) and input_df is not None:
        processed = preprocess_input(input_df)
        X_scaled = scaler.transform(processed)
        preds = model.predict(X_scaled)
        
        if mode == "Ввести вручную":
            st.success(f"Прогноз: {preds[0]:,.0f} ₽")
        else:
            input_df['Predicted_Price'] = preds
            st.dataframe(input_df.head())
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV", csv, "predictions.csv", "text/csv")

# === ВКЛАДКА 3: ВЕСА МОДЕЛИ ===
with tab3:
    st.header("Интерпретация модели")
    coefs = model.coef_
    feats = ohe_columns
    weights = pd.DataFrame({'Feature': feats, 'Weight': coefs})
    weights['Abs_Weight'] = weights['Weight'].abs()
    weights = weights.sort_values(by='Abs_Weight', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=weights, x='Weight', y='Feature', hue='Feature', legend=False, palette='viridis', ax=ax)
    plt.title("Топ-15 признаков")
    st.pyplot(fig)