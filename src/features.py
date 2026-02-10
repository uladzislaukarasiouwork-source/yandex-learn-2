import pandas as pd
from datetime import datetime


def add_time_features(df):
    """
    Добавляет временные признаки на основе столбца pickup_datetime
    Преобразует pickup_datetime в datetime, извлекает час и день недели
    Возвращает DataFrame с новыми признаками и удаленным исходным столбцом
    Нет проверки наличия столбца 'pickup_datetime'
    """
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df.drop('pickup_datetime', axis=1)
