import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter

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

# Train the k-NN algorithm
correct = 0
for index1, row in X_train.iterrows():
    print(y_train[index1])
    x = row.values
    distance = [np.sqrt(np.sum((row.values - x) ** 2)) for index2, row in X_train.iterrows()]
    idx_sort = np.argsort(distance)
    y_idx_sort = y_train.values[idx_sort]
    topK_y = y_idx_sort[:112]
    c = Counter(topK_y)
    print(c.most_common()[0][0])
    if c.most_common()[0][0] == y_train[index1]:
        correct += 1
