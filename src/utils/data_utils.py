import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocessing(data, standardize: bool = False):
    if standardize:
        scaler = StandardScaler
    else:
        scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def load_data():
    df = pd.read_csv("data\dataset_200x4_regression.csv", sep=',')
    x, t = df.iloc[:, :3], df.iloc[:, 3]
    x = preprocessing(x)
    return x, t