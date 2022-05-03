import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def f1_score(actual, predicted):
    TP = np.sum(np.multiply([i == True for i in predicted], actual))
    TN = np.sum(np.multiply([i == False for i in predicted], [not j for j in actual]))
    FP = np.sum(np.multiply([i == True for i in predicted], [not j for j in actual]))
    FN = np.sum(np.multiply([i == False for i in predicted], actual))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if precision != 0 and recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


def minmax_scale(df_in):
    df_norm = (df_in - df_in.min()) / (df_in.max() - df_in.min())
    return df_norm


if __name__ == '__main__':
    # Load data
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    digits_dataset = np.c_[digits_dataset_X, digits_dataset_y.T]
    df = pd.DataFrame(digits_dataset)
    col_class = df.pop(64)
    df = minmax_scale(df)
    df = df.drop(df.columns[[0, 32, 39]], axis=1)
    df.insert(len(df.columns), 64, col_class)

    list_target = df[64].unique()
    df9 = df[df[64].isin([list_target[0]])]
    df8 = df[df[64].isin([list_target[1]])]
    df7 = df[df[64].isin([list_target[2]])]
    df6 = df[df[64].isin([list_target[3]])]
    df5 = df[df[64].isin([list_target[4]])]
    df4 = df[df[64].isin([list_target[5]])]
    df3 = df[df[64].isin([list_target[6]])]
    df2 = df[df[64].isin([list_target[7]])]
    df1 = df[df[64].isin([list_target[8]])]
    df0 = df[df[64].isin([list_target[9]])]
    # Split into folds
    k_fold = []
    fold_size9 = int(len(df9) / 10)
    fold_size8 = int(len(df8) / 10)
    fold_size7 = int(len(df7) / 10)
    fold_size6 = int(len(df6) / 10)
    fold_size5 = int(len(df5) / 10)
    fold_size4 = int(len(df4) / 10)
    fold_size3 = int(len(df3) / 10)
    fold_size2 = int(len(df2) / 10)
    fold_size1 = int(len(df1) / 10)
    fold_size0 = int(len(df0) / 10)
    for k in range(0, 9):
        fold9 = df9.sample(n=fold_size9, random_state=587)
        fold8 = fold9.append(df8.sample(n=fold_size8, random_state=587))
        fold7 = fold8.append(df7.sample(n=fold_size7, random_state=587))
        fold6 = fold7.append(df6.sample(n=fold_size6, random_state=587))
        fold5 = fold6.append(df5.sample(n=fold_size5, random_state=587))
        fold4 = fold5.append(df4.sample(n=fold_size4, random_state=587))
        fold3 = fold4.append(df3.sample(n=fold_size3, random_state=587))
        fold2 = fold3.append(df2.sample(n=fold_size2, random_state=587))
        fold1 = fold2.append(df1.sample(n=fold_size1, random_state=587))
        fold0 = fold1.append(df0.sample(n=fold_size0, random_state=587))
        df9 = df9[~df9.index.isin(fold9.index)]
        df8 = df8[~df8.index.isin(fold8.index)]
        df7 = df7[~df7.index.isin(fold7.index)]
        df6 = df6[~df6.index.isin(fold6.index)]
        df5 = df5[~df5.index.isin(fold5.index)]
        df4 = df4[~df4.index.isin(fold4.index)]
        df3 = df3[~df3.index.isin(fold3.index)]
        df2 = df2[~df2.index.isin(fold2.index)]
        df1 = df1[~df1.index.isin(fold1.index)]
        df0 = df0[~df0.index.isin(fold0.index)]
        k_fold.append(fold0)
    k_fold.append(df9.append(df8.append(df7.append(df6.append(df5.append(df4.append(df3.append(
        df2.append(df1.append(df0))))))))))

    k = 11
    fold_idx = 5

    k_fold_copy = k_fold.copy()
    data_test = k_fold[fold_idx]
    del k_fold_copy[fold_idx]
    data_train = pd.concat(k_fold_copy).sample(n=len(df) - len(data_test.index), replace=True, random_state=587)
    data_train = data_train.drop(data_train.index[0], axis=0)

    score = []
    for n_sample in range(0, len(data_train), 10):
        # Split to train and test dataset
        k_fold_copy = k_fold.copy()
        data_test = k_fold[fold_idx]
        del k_fold_copy[fold_idx]
        data_train = pd.concat(k_fold_copy).sample(n=len(df) - len(data_test.index), replace=True, random_state=587)
        X_train = data_train.drop(64, axis=1).values
        X_train = np.delete(X_train, range(0, n_sample), axis=0)
        y_train = data_train[64].values.astype(int)
        y_train = np.delete(y_train, range(0, n_sample), axis=0)
        X_test = data_test.drop(64, axis=1).values
        y_test = data_test[64].values.astype(int)
        print('Training set size:', len(X_train))

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train.tolist())
        y_test = pd.Series(y_test.tolist())

        y_pred = []
        for index1, row1 in X_test.iterrows():
            x = row1.values
            distance = [np.sqrt(np.sum((row2.values - x) ** 2)) for index2, row2 in X_train.iterrows()]
            idx_sort = np.argsort(distance)
            y_idx_sort = y_train.values[idx_sort]
            y_top_k = y_idx_sort[:k]
            pred = max(set(y_top_k), key=y_top_k.tolist().count)
            y_pred.append(pred)
        y_test = y_test.tolist()
        f1_i = f1_score(y_test, y_pred)
        accuracy_i = np.sum(np.array(y_pred) == np.array(y_test)) / len(y_test)
        score.append(1-accuracy_i)
        print('n_sample:', n_sample, 'score:', 1-accuracy_i)

    plt.plot(range(0, len(data_train), 10), sorted(score, reverse=True),
             '.-', markersize=10, color='#1f77b4')
    plt.xlabel('Number of training samples')
    plt.ylabel('Cost function')
    plt.title('The Hand-Written Dataset')
    plt.savefig("learning_curve_hand.eps", dpi=600, format="eps")
    plt.show()

