def add_features(df):
    df["credit_per_month"] = df["credit_amount"] / df["duration_months"]
    return df