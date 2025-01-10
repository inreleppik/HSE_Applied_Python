import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def weather_analysis(city_data):
    data = city_data.sort_values(by='timestamp').copy()
    city = data.city.values[0]
    mean_temp = data.temperature.mean()
    min_temp = data.temperature.min()
    max_temp = data.temperature.max()
    r_mean = data.temperature.rolling(window=30).mean()
    r_std = data.temperature.rolling(window=30).std()
    anom = data[(data.temperature > r_mean + 2*r_std) | (data.temperature < r_mean - 2*r_std)][['timestamp','temperature']]
    season_st = data.groupby('season', as_index=False).agg(mean_temp=('temperature','mean'), std_temp=('temperature','std'))
    y = data.temperature.values
    model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=365)
    fit = model.fit()
    trend_values = fit.fittedvalues
    seasonal_values = fit.season
    slope = fit.params['smoothing_trend']
    return {
        'city': city,
        'mean_temp': mean_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'season_st': season_st,
        'trend_values': trend_values,
        'seasonal_values': seasonal_values,
        'slope': slope,
        'anomalies': anom
    }

def get_temp_sync(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    if response.status_code == 200:
        return data["main"]["temp"]
    elif response.status_code == 401:
        st.error(f"{data}")
        return None
    else:
        st.error(f"Ошибка для города {city}: {data.get('message', 'Unknown error')}")
        return None

def get_season(date):
    if date.month in [12, 1, 2]:
        return "winter"
    elif date.month in [3, 4, 5]:
        return "spring"
    elif date.month in [6, 7, 8]:
        return "summer"
    elif date.month in [9, 10, 11]:
        return "autumn"

def is_anomaly(temp, hist_data, city, season=None):
    if season is None:
        season = get_season(datetime.now())
    stats = hist_data[city]['season_st']
    season_stats = stats[stats.season == season]
    mean_s = season_stats.mean_temp.values[0]
    std_s = season_stats.std_temp.values[0]
    if (temp > (mean_s + 2*std_s)) or (temp < (mean_s - 2*std_s)):
        return True
    else:
        return False

st.title('Анализ температурных данных и мониторинг текущей температуры')
st.header('Загрузка исторических данных')

uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    hist_data = Parallel(n_jobs=-1)(
        delayed(weather_analysis)(data[data.city == i]) 
        for i in data.city.unique().tolist()
    )
    hist_data = {i: result for i, result in zip(data.city.unique().tolist(), hist_data)}
    st.write('Данные успешно загружены')
    st.header('Выберите интересующий вас город')
    selected_city = st.selectbox('Выберите вариант:', data.city.unique().tolist())
    stats = hist_data[selected_city]
    data_city = data[data.city == selected_city]
    st.write(f'Вами был выбран следущий город: {selected_city}')
    st.header('API OpenWeatherMap')
    API_KEY = st.text_input('Введите ваш API-ключ:')
    if API_KEY and uploaded_file:
        st.header('Получение текущей температуры')
        current_temp = get_temp_sync(selected_city)
        if current_temp is not None:
            st.success(f"Текущая температура в городе {selected_city}: {current_temp} °C")
            anomaly = is_anomaly(current_temp, hist_data, selected_city)
            if anomaly:
                st.warning("Текущая температура аномальна!")
            else:
                st.success("Текущая температура в пределах нормы.")
    if st.checkbox('Показать описательную статистику'):
        st.header('Описательная статистика')
        st.write(f"Средняя температура в {selected_city}: {stats['mean_temp']:.2f} °C")
        st.write(f"Минимальная температура в {selected_city}: {stats['min_temp']:.2f} °C")
        st.write(f"Максимальная температура в {selected_city}: {stats['max_temp']:.2f} °C")
    if st.checkbox('Показать сезонный профиль'):
        st.header(f'Сезонный профиль в городе {selected_city}')
        st.dataframe(stats['season_st'])
    if st.checkbox('Показать график временного ряда'):
        seasonal_values = stats['seasonal_values']
        trend_values = stats['trend_values']
        anomalies = stats['anomalies']
        st.header("График временного ряда с выделенными аномалиями, сезонностью и трендом")
        start_date, end_date = st.slider(
            "Выберите временной диапазон:",
            min_value=data_city["timestamp"].min().to_pydatetime(),
            max_value=data_city["timestamp"].max().to_pydatetime(),
            value=(
                data_city["timestamp"].min().to_pydatetime(),
                data_city["timestamp"].max().to_pydatetime()
            ),
            format="YYYY-MM-DD",
        )
        f_dc = data_city[(data_city["timestamp"] >= start_date) & (data_city["timestamp"] <= end_date)]
        f_a = anomalies[(anomalies["timestamp"] >= start_date) & (anomalies["timestamp"] <= end_date)]
        f_tv = trend_values[:len(f_dc)]
        f_sv = seasonal_values[:len(f_dc)]
        fig = px.line(
            f_dc,
            x="timestamp",
            y="temperature",
            color="city",
            line_group="city",
            title="Температура и выделенные аномалии",
            labels={"temperature": "Temperature (°C)", "timestamp": "Date"},
        )
        fig.add_scatter(
            x=f_a['timestamp'],
            y=f_a['temperature'],
            mode='markers',
            marker=dict(color='red', size=6, symbol='circle'),
            name='Anomalies'
        )
        fig.add_scatter(
            x=f_dc['timestamp'],
            y=f_tv,
            mode='lines',
            name='Trend',
            line=dict(color='orange', dash='dash'),
        )
        fig.add_scatter(
            x=f_dc['timestamp'],
            y=f_sv,
            mode='lines',
            name='Season',
            line=dict(color='purple'),
        )
        fig.update_layout(width=800, height=700)
        st.plotly_chart(fig)
else:
    st.write("Пожалуйста загрузите свои данные")
