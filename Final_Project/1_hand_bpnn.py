import pandas as pd
from sklearn import datasets
import numpy as np


def minmax_scale(df_in):
    df_norm = (df_in - df_in.min()) / (df_in.max() - df_in.min())
    return df_norm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(h):
    return h * (1 - h)


def f1_score(actual, predicted):
    TP = np.sum(np.multiply([i == True for i in predicted], actual))
    TN = np.sum(np.multiply([i == False for i in predicted], [not j for j in actual]))
    FP = np.sum(np.multiply([i == True for i in predicted], [not j for j in actual]))
    FN = np.sum(np.multiply([i == False for i in predicted], actual))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class BPNNClassifier:
    def __init__(self, in_n, hid_l=1, hid_n=4, out_n=2, eta=0.1, lmbda=0.02, max_iter=200):
        self.in_n = in_n
        self.hid_l = hid_l + 1
        self.hid_n = hid_n
        self.out_n = out_n
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.weights = []
        self.grad = list(range(self.hid_l))  # save the gradient of every neuron
        self.values = []  # save the activated value of every neuron

        for d in range(self.hid_l):
            if d == 0:  # input layer to hidden layer
                weight = np.random.randn(self.hid_n, in_n + 1)
            elif d == self.hid_l - 1:  # hidden layer to output layer
                out_n = 1 if out_n == 2 else out_n
                weight = np.random.randn(out_n, self.hid_n)
            else:  # the others
                out_n = 1 if out_n == 2 else out_n
                weight = np.random.randn(self.hid_n, self.hid_n)
            self.weights.append(weight)

    def linear_input(self, hid_l, x):
        weight = self.weights[hid_l]
        return x @ weight.T

    def preprocessing(self, x=None, y=None):
        x_y = []
        if isinstance(x, np.ndarray):
            x0 = np.array([[1] for _ in range(x.shape[0])])
            x = np.hstack([x0, x])
            x_y.append(x)
        if isinstance(y, np.ndarray):
            y = self.encoder(y)
            x_y.append(y)
        return tuple(x_y)

    def encoder(self, y):
        y_new = []
        if y.ndim == 1:  # encode y to one hot code
            if self.out_n > 2:
                for yi in y:
                    yi_new = np.zeros(self.out_n)
                    yi_new[yi] = 1
                    y_new.append(yi_new)
                y_new = np.array(y_new)
            else:
                y_new = y
        elif y.ndim == 2:  # encode y to 1D array
            if self.out_n > 2:
                for yi in y:
                    for j in range(len(yi)):
                        if yi[j] == 1:
                            y_new.append(j)
                            break
                y_new = np.array(y_new)
            else:
                y_new = y.ravel()
        else:
            raise Exception("argument value error: ndarray ndim should be 1 or 2")
        return y_new

    def forward_propagation(self, x):
        self.values.clear()
        value = None
        for d in range(self.hid_l):
            if d == 0:  # input layer to hidden layer
                value = sigmoid(self.linear_input(d, x))
            elif d == self.hid_l - 1:  # hidden layer to output layer, use sigmoid
                value = sigmoid(self.linear_input(d, value))
            else:  # the others
                value = sigmoid(self.linear_input(d, value))
            self.values.append(value)
        return value

    def back_propagation(self, y_true):
        for d in range(self.hid_l - 1, -1, -1):
            if d == self.hid_l - 1:  # hidden layer to output layer
                self.grad[d] = (y_true - self.values[d]) * d_sigmoid(self.values[d])
            else:
                self.grad[d] = self.grad[d + 1] @ self.weights[d + 1] * d_sigmoid(self.values[d])

    def standard_bp(self, x, y):
        for _ in range(self.max_iter):
            for xi, yi in zip(x, y):
                # forward propagation
                self.forward_propagation(xi)
                # back propagation
                self.back_propagation(yi)
                # update weight
                for d in range(self.hid_l):
                    if d == 0:  # input layer to hidden layer
                        self.weights[d] += self.grad[d].reshape(-1, 1) @ xi.reshape(1, -1) * self.eta * (1 - self.lmbda)
                    else:  # the others
                        self.weights[d] += self.grad[d].reshape(-1, 1) @ \
                                           self.values[d - 1].reshape(1, -1) * self.eta * (1 - self.lmbda)

    def fit(self, x, y):
        x, y = self.preprocessing(x, y)
        self.standard_bp(x, y)
        return self

    def predict(self, x, probability=False):
        x = self.preprocessing(x)[0]
        prob = self.forward_propagation(x)
        if self.out_n == 2:  # binary classification
            y = np.where(prob >= 0.5, 1, 0)
        else:  # multiply classification
            y = np.zeros(prob.shape)
            for yi, i in zip(y, np.argmax(prob, axis=1)):
                yi[i] = 1
        y = self.preprocessing(y=y)[0]
        if probability:
            return y, prob
        else:
            return y


if __name__ == "__main__":
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

    fold_idx = 0
    accuracy_k = []
    f1_k = []
    while fold_idx < 10:
        print('kfold index: ', fold_idx)
        # Split to train and test dataset
        k_fold_copy = k_fold.copy()
        data_test = k_fold[fold_idx]
        del k_fold_copy[fold_idx]
        data_train = pd.concat(k_fold_copy).sample(n=len(df) - len(data_test.index), replace=True, random_state=587)
        X_train = data_train.drop(64, axis=1).values
        y_train = data_train[64].values.astype(int)
        X_test = data_test.drop(64, axis=1).values
        y_test = data_test[64].values.astype(int)

        # Train the model and predict
        classifier = BPNNClassifier(in_n=61, hid_l=8, hid_n=16, out_n=10, lmbda=0.05).fit(X_train, y_train)
        prediction = classifier.predict(X_test)

        final_true = y_test.tolist()
        final_prediction = prediction.tolist()

        f1_i = f1_score(final_true, final_prediction)
        accuracy_i = np.sum(np.array(final_prediction) == np.array(final_true)) / len(final_true)
        f1_k.append(f1_i)
        accuracy_k.append(accuracy_i)
        fold_idx += 1
    print("Accuracy:", np.mean(accuracy_k))
    print("F1:", np.mean(f1_k))
