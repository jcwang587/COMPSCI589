import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
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

        # Train the k-NN algorithm using training set
        accuracy = []
        for k in k_list:
            correct = 0
            for index1, row1 in X_train.iterrows():
                x = row1.values
                distance = [np.sqrt(np.sum((row2.values - x) ** 2)) for index2, row2 in X_train.iterrows()]
                idx_sort = np.argsort(distance)
                y_idx_sort = y_train.values[idx_sort]
                y_top_k = y_idx_sort[:k]
                if max(y_top_k.tolist(), key=y_top_k.tolist().count) == y_train[index1]:
                    correct += 1
            accuracy.append(correct / len(y_train))

        # Compute the accuracy of the k-NN model making predictions for training set
        final_accuracy[i] = pd.DataFrame(accuracy)

    final = pd.concat(list(final_accuracy.values()), axis=1)
    std = np.std(final, axis=1)
    avg = np.mean(final, axis=1)
    plt.errorbar(k_list, avg, yerr=std, fmt="-", ecolor="red", elinewidth=1, capsize=5, capthick=2)
    plt.plot(k_list, avg, '.', markersize=10, color='#1f77b4')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy over training data')
    plt.savefig("FigurekNN1.eps", dpi=600, format="eps")
