import pandas as pd

df = pd.read_csv("models/clean_data.csv")
print(df[['ImageURL']].head(100))  # Check if ImageURL exists
