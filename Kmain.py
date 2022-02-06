import pandas as pd
import numpy as np
import operator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter


def euclidean_distance(data_1, data_2, data_len):
    dist = 0
    for i in range(data_len):
        dist = dist + np.square(data_1[i] - data_2[i])
    return np.sqrt(dist)


def knn(dataset, test_instance, k):
    distances = {}
    length = test_instance.shape[1]
    for x in range(len(dataset)):
        dist_up = euclidean_distance(test_instance, dataset.iloc[x], length)
        distances[x] = dist_up[0]
    # Sort values based on distance
    sort_distances = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    # Extracting nearest k neighbors
    for x in range(k):
        neighbors.append(sort_distances[x][0])
    # Initializing counts for 'class' labels counts as 0
    counts = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
    # Computing the most frequent class
    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1]
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
    # Sorting the class in reverse order to get the most frequent class
    sort_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return sort_counts[0][0]


# Load the dataset
df = pd.read_csv('iris.csv', header=None)
df.columns = list(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Shuffle the dataset
df_sf = shuffle(df, random_state=0)
X = df_sf[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df_sf['species']

# Normalize the dataset
X = (X - X.min()) / (X.max() - X.min())

# Randomly partition the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a list of all columns except 'class' by iterating through the development set
row_list = []
for index, rows in X_test.iterrows():
    my_list = [rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]
    row_list.append([my_list])
# k values for the number of neighbors that need to be considered
k_n = [1, 3, 5, 7]
# Distance metrics
# Performing kNN on the development set by iterating all the development set data points and for each k and each
# distance metric
development_set_obs_k = {}
for k in k_n:
    development_set_obs = []
    for i in range(len(row_list)):
        development_set_obs.append(
            knn(X_train, pd.DataFrame(row_list[i]), k))
    development_set_obs_k[k] = development_set_obs
    # Nested Dictionary containing the observed class for each k and each distance metric (obs_k of the form obs_k[
    # dist_method][k])
    obs_k = development_set_obs_k
# print(obs_k)


# # Load the dataset
# df = pd.read_csv('iris.csv', header=None)
# df.columns = list(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
#
# # Shuffle the dataset
# df_sf = shuffle(df, random_state=0)
# X = df_sf[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# y = df_sf['species']
#
# # Normalize the dataset
# X = (X - X.min()) / (X.max() - X.min())
#
# # Randomly partition the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the k-NN algorithm
# correct = 0
# for index1, row in X_train.iterrows():
#     print(y_train[index1])
#     x = row.values
#     distance = [np.sqrt(np.sum((row.values - x) ** 2)) for index2, row in X_train.iterrows()]
#     distance.remove(0)
#     idx_sort = np.argsort(distance)
#     y_idx_sort = y_train.values[idx_sort]
#     topK_y = y_idx_sort[:70]
#     c = Counter(topK_y)
#     print(c.most_common()[0][0])
#     if c.most_common()[0][0] == y_train[index1]:
#         correct += 1
