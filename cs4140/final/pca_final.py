import pandas as pd
import numpy as np

df = pd.read_csv("movie_data_all_rows_complete.csv")

df = df[df['budget'] > 0]

# Add score column
df['score'] = ((df['revenue'] - df['budget']) / df['budget']) * ((df['vote_average'] * df['popularity']) / 2)

# Step 1: Sort the DataFrame by budget
df_sorted = df.sort_values(by='budget').reset_index(drop=True)

# Step 2: Mark first 7500 as 'Low Budget', the rest as 'High Budget'
df_sorted['budget_type'] = ['Low Budget' if i < 7500 else 'High Budget' for i in range(len(df_sorted))]

df['budget_type'] = df_sorted['budget_type'].values

numerics = [
    'budget',
    'popularity',
    'revenue',
    'runtime',
    'vote_average',
    'score',  # new column you added
]
genre_cols = [col for col in df.columns if col.startswith('genres_')]

features = numerics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print(f"{'Genre':<20} {'Explained Var PC1':>18} {'Explained Var PC2':>18}{'Total':>18}")

for genre_col in genre_cols:
    genre_name = genre_col.replace("genres_", "")
    genre_df = df[df[genre_col] == True]
    
    if len(genre_df) < 2:
        continue  # Skip genres with too few movies

    X_genre = genre_df[features].copy()
    X_scaled_genre = StandardScaler().fit_transform(X_genre)

    pca_genre = PCA(n_components=2)
    principal_components_genre = pca_genre.fit_transform(X_scaled_genre)

    explained_var = pca_genre.explained_variance_ratio_
    print(f"{genre_name:<20} {explained_var[0]:18.3f} {explained_var[1]:18.3f}{explained_var[1]+explained_var[0]:18.3f}")



X = df[features].copy()
X_scaled = StandardScaler().fit_transform(X)


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['budget_type'] = df['budget_type']


# PCA object already fitted as `pca`
loadings = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
print(loadings.T.sort_values(by='PC1', ascending=False))  # PC1 influence
print(loadings.T.sort_values(by='PC2', ascending=False))  # PC2 influence

df_sorted_by_score = df.sort_values(by='score', ascending=False)

print(loadings.head())

# Define success based on top 10% of score
score_threshold = df['score'].quantile(0.99)
df_pca['successful'] = df['score'] >= score_threshold

import matplotlib.pyplot as plt

# Define markers for budget types
markers = {'Low Budget': '^', 'High Budget': 's'}
color = {'Low Budget': 'green', 'High Budget': 'blue'}

# Filter only successful movies
successful_movies = df_pca[df_pca['successful'] == True]

plt.figure(figsize=(10, 7))

# Plot successful movies with different markers based on budget type
for budget_type in successful_movies['budget_type'].unique():
    subset = successful_movies[successful_movies['budget_type'] == budget_type]
    plt.scatter(
        subset['PC1'],
        subset['PC2'],
        marker=markers[budget_type],
        color=color[budget_type],
        alpha=0.75,
        label=budget_type
    )

# Labels and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Successful Movies by Budget Type in PCA Space')
plt.legend(title='Budget Type')
plt.grid(True)
plt.tight_layout()
plt.show()


"""import matplotlib.pyplot as plt

# Define styles
markers = {'Low Budget': '^', 'High Budget': 's'}
colors = {True: 'green', False: 'gray'}

plt.figure(figsize=(10, 7))

# Loop through combinations of budget_type and success
for budget_type in df_pca['budget_type'].unique():
    for success in [True, False]:
        subset = df_pca[(df_pca['budget_type'] == budget_type) & (df_pca['successful'] == success)]
        plt.scatter(
            subset['PC1'],
            subset['PC2'],
            marker=markers[budget_type],
            color=colors[success],
            alpha=0.7,
            label=f"{budget_type} - {'Successful' if success else 'Other'}"
        )

# Labels and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Movies: Budget Type and Success')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()"""

"""
import matplotlib.pyplot as plt

colors = {'Low Budget': 'blue', 'High Budget': 'red'}

plt.figure(figsize=(8, 6))
for budget_type in colors:
    indices = df_pca['budget_type'] == budget_type
    plt.scatter(
        df_pca.loc[indices, 'PC1'],
        df_pca.loc[indices, 'PC2'],
        c=colors[budget_type],
        label=budget_type,
        alpha=0.6
    )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Movies Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

