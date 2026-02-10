import pandas as pd

def load_data(path):
    """
    Загружает данные из CSV-файла
    Принимает путь к файлу, возвращает pandas DataFrame
    Необходимо добавить docstring с описанием параметров, возвращаемого значения и возможных исключений
    """
    return pd.read_csv(path)

def split_data(data):
    """
    Разделяет данные на обучающую и тестовую выборки
    Предполагается, что в данных есть колонка 'fare_amount' (Она там есть!)
    Возвращает кортеж: (X_train, X_test, y_train, y_test)
    Где X - признаки (все колонки кроме 'fare_amount'), y - целевая переменная ('fare_amount')
    """
    from sklearn.model_selection import train_test_split
    """
    Перенести импорт в начало файла к pandas
    """
    features = data.drop('fare_amount', axis=1)
    target = data['fare_amount']
    return train_test_split(features, target, test_size=0.2)
    """
    Непонятно, что возвращает функция и какие параметры использует
    Добавь docstring с описанием возвращаемых значений
    """
