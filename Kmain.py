import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter

# Load the dataset
df = pd.read_csv('iris.csv', header=None)
df.columns = list(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Train the k-NN algorithm
loop = 20
k_list = range(1, 120, 10)
final_accuracy = {}
for i in range(0, loop):
    # Shuffle the dataset
    df_sf = shuffle(df)
    X = df_sf[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df_sf['species']

    # Randomly partition the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalize the dataset
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    accuracy = []
    for k in k_list:
        correct = 0
        for index, row in X_train.iterrows():
            x = row.values
            distance = [np.sqrt(np.sum((row.values - x) ** 2)) for index2, row in X_train.iterrows()]
            idx_sort = np.argsort(distance)
            y_idx_sort = y_train.values[idx_sort]
            topK_y = y_idx_sort[:k]
            c = Counter(topK_y)
            if c.most_common()[0][0] == y_train[index]:
                correct += 1
        accuracy.append(correct / 120)
    final_accuracy[i] = pd.DataFrame(accuracy)

final = pd.concat(list(final_accuracy.values()), axis=1)
std = np.std(final, axis=1)
avg = np.mean(final, axis=1)
# plt.plot(k_list, accuracy, 'bo-')
plt.errorbar(k_list, avg, yerr=std, fmt="-", ecolor="red", elinewidth=0.5, capsize=2, capthick=1)
plt.title('The Lasers in Three Conditions')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
