import pandas as pd

df = pd.read_csv("movie_data_all_rows_complete.csv")

df = df.drop(columns=[col for col in df.columns if col.startswith('genres_')])
df = df.drop(columns=["tagline", "production_countries", "production_companies", "overview", "original_language"])
df_sorted = df.sort_values(by='budget').reset_index(drop=True)

idx = [3757,6864, 7915, 7674, 128,7910]
selected = df_sorted.loc[idx]

print(selected.sort_values(by="budget"))