import pandas as pd

def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    return X, y


