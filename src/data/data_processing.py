
def clean_and_prepare_data(df):
    df = df.dropna()
    df["risk"] = df["risk"].map({1: 0, 2: 1})
    
    return df