import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocessing(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform()

def load_data():
    df = pd.read_csv("data\dataset_200x4_regression.csv", sep=',')
    x, t = df.iloc[:, :3], df.iloc[:, 3]
    x = preprocessing(x)
    return x, t