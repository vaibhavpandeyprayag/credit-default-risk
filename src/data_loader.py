import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    return df

def load_csv_data(path):
    df = pd.read_csv(path)
    return df