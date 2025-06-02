import pandas as pd

# import the dataset into a pandas dataframe

movie_df = pd.read_csv("movie_dataset.csv")
print(movie_df.head())

# view type of data in the dataframe
print(movie_df.dtypes)

### normalize date time columns
movie_df["release_date"] = pd.to_datetime(movie_df["release_date"], format="%Y-%m-%d", errors="coerce")

print("There are a total of ", movie_df["release_date"].isna().sum(), " invalid dates")

### cast all numeric columns to floats
numeric_columns = [
    'movie_id',
    'budget',
    'popularity',
    'revenue',
    'runtime',
    'vote_average'
]

movie_df[numeric_columns] = movie_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

print(movie_df.dtypes)

### Create a one hot encoding for genres column
movie_df["genres"] = movie_df["genres"].str.split(", ")
movie_df = movie_df.explode("genres")
movie_df = pd.get_dummies(movie_df, columns=["genres"])
movie_df = movie_df.groupby("movie_id", as_index=False).max()

### Make original_language categorical as there are too many fields to use one-hot
movie_df["original_language"] = movie_df["original_language"].astype("category")

### May want to explode production_companies and production_countries but am not going to do this now.


# Remove all rows with budget, revenue, or runtime listed as 0
movie_df = movie_df[(movie_df["budget"] != 0) & (movie_df["revenue"] != 0) & (movie_df["runtime"] != 0) &
                    (movie_df["budget"].notna()) & (movie_df["revenue"].notna()) & (movie_df["runtime"].notna())]
movie_df = movie_df.dropna(how='any')
movie_df = movie_df[(movie_df != '').all(axis=1)].dropna(how='any')



# save back to csv file
movie_df.to_csv("movie_data_all_rows_complete.csv", index=False)