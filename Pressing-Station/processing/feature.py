def make_features(df):
    df = df.set_index("timestamp")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[f'roll_mean_5_{col}'] = df[col].rolling(5).mean()
    return df.dropna().reset_index()

def save_features(df, path):
    df.to_csv(path, index=False)
