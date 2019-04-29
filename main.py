import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,     

def prep100k():
    col_data = ['user_id', 'movie_id', 'rating_user', 'timestamp_user']

    data = pd.read_csv("./datasets/ml-100k/u.data",
                       names=col_data,  delimiter="\t")

    col_item_data = ['movie_id', 'movie_title', 'release date', 'video_release_date',
                     'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi',
                     'Thriller', 'War', 'Western']

    itemData = pd.read_csv("./datasets/ml-100k/u.item",  names=col_item_data, encoding="ISO-8859-1",
                           delimiter="|")
    dataset = pd.merge(data, itemData, on='movie_id', how='outer')

    return dataset


def prep1m():
    col_data = ['user_id', 'movie_id', 'rating_user', 'timestamp_user']

    data = pd.read_csv("./datasets/ml-1m/ratings.dat",
                       names=col_data,  delimiter="::", engine='python')

    col_item_data = ['movie_id', 'movie_title', 'categories']

    itemData = pd.read_csv("./datasets/ml-1m/movies.dat",  names=col_item_data, encoding="ISO-8859-1",
                           delimiter="::", engine='python')

    dataset = pd.merge(data, itemData, on='movie_id', how='outer')

    return dataset

np.random.seed(5)

dataset = prep100k()

nusers = 1682
nmovies = 943

# Cria array de dados com ? onde não há dados
data = [['?' for x in range(nusers)] for y in range(nmovies)]

# Preenche o array com os dados da tabela
for row_index, row in dataset.iterrows():
    data[row['user_id']-1][row['movie_id']-1] = row['rating_user']

# Calcula a média de cada linha e preenche com a média os valores '?' na tabela
for row in data:
        total = 0
        count = 0
        for col in row:
            if col != '?':
                total += col
                count += 1
        avg = 0
        if count > 0:
            avg = total/count
        for i, val in enumerate(row):
            if val == '?':
                row[i] = avg


mldataarray = np.asarray(data)

kmeans = KMeans(n_clusters=17, random_state=0).fit(mldataarray)

print('Star Wars: Episode I')
print(kmeans.predict(mldataarray[5].reshape(1,-1)))
print('Star Wars: Episode IV')
print(kmeans.predict(mldataarray[4].reshape(1,-1)))
print('Star Wars: Episode V')
print(kmeans.predict(mldataarray[50].reshape(1,-1)))
print('Star Wars: Episode VI')
print(kmeans.predict(mldataarray[60].reshape(1,-1)))