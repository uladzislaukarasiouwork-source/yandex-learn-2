import os
import pandas as pd
from src.data import load_data, split_data
from src.features import add_time_features
from src.model import TaxiFareModel

"""
Глобальная переменная DATA_PATH -- проблема в разработке
Нужно сделать параметром скрипта или вынести в конфигурационный файл
Также не существует data/main.csv 
Наш путь data/uber.csv
"""
DATA_PATH = "data/main.csv"

# Загрузка и обработка данных
raw_data = load_data(DATA_PATH)
"""
Необходима обработка ошибок загрузки файла
Нет проверки, что данные загрузились корректно
"""
processed_data = add_time_features(raw_data)
"""
Нет логирования этапов пайплайна а так же нет вывода информации о разделении данных
"""
X_train, X_test, y_train, y_test = split_data(processed_data)

# Обучение модели
model = TaxiFareModel()
"""
Нельзя настроить гиперпараметры модели! Обратить внимание на файл src/model.py
"""
model.fit(X_train, y_train)
"""
Нет сохранения обученной модели. Необходимо добавить!
"""

# Оценка
"""
Заменить на model.score(X_test, y_test)
"""
score = model.model.score(X_test, y_test)
"""
необходимо добавить больше метрик! А так же дать возможность сохранять результаты
"""
print(f"R²: {score:.2f}")
