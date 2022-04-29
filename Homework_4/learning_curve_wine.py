import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(h):
    return h * (1 - h)


def accuracy_score(y_true, y_pred):
    score = y_true == y_pred
    return np.average(score)


def precision_score(y_true, y_pred):
    tp_tn_idx = np.where(y_true == y_pred)[0].tolist()
    tp = [y_pred[i] for i in tp_tn_idx].count(1)
    tp_fp = y_pred.count(1)
    return tp / tp_fp


def recall_score(y_true, y_pred):
    tp_tn_idx = np.where(y_true == y_pred)[0].tolist()
    tp = [y_pred[i] for i in tp_tn_idx].count(1)
    tp_fn = y_true.tolist().count(1)
    return tp / tp_fn


def f1_score(precision_value, recall_value):
    return 2 * precision_value * recall_value / (precision_value + recall_value)


class BPNNClassifier:
    def __init__(self, feature_n, hidden_n=10, deep=2, label_n=2, eta=0.1, max_iter=200, lambd=0):
        self.feature_n = feature_n
        self.hidden_n = hidden_n
        self.deep = deep
        self.label_n = label_n
        self.eta = eta
        self.max_iter = max_iter
        self.weights = []
        self.grad = list(range(deep))  # save the gradient of every neuron
        self.values = []  # save the activated value of every neuron

        for d in range(deep):
            if d == 0:  # input layer to hidden layer
                weight = np.random.randn(hidden_n, feature_n + 1)
            elif d == self.deep - 1:  # hidden layer to output layer
                label_n = 1 if label_n == 2 else label_n
                weight = np.random.randn(label_n, hidden_n)
            else:  # the others
                label_n = 1 if label_n == 2 else label_n
                weight = np.random.randn(hidden_n, hidden_n)
            self.weights.append(weight)

    def linear_input(self, deep, x):
        weight = self.weights[deep]
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
            if self.label_n > 2:
                for yi in y:
                    yi_new = np.zeros(self.label_n)
                    yi_new[yi] = 1
                    y_new.append(yi_new)
                y_new = np.array(y_new)
            else:
                y_new = y
        elif y.ndim == 2:  # encode y to 1D array
            if self.label_n > 2:
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

    def forward_propagation(self, x, lambd=0):
        self.values.clear()
        value = None
        for d in range(self.deep):
            if d == 0:  # input layer to hidden layer
                value = sigmoid(self.linear_input(d, x)) + lambd
            elif d == self.deep - 1:  # hidden layer to output layer, use sigmoid
                value = sigmoid(self.linear_input(d, value)) + lambd
            else:  # the others
                value = sigmoid(self.linear_input(d, value)) + lambd
            self.values.append(value)
        return value

    def back_propagation(self, y_true):
        for d in range(self.deep - 1, -1, -1):
            if d == self.deep - 1:  # hidden layer to output layer
                self.grad[d] = (y_true - self.values[d]) * d_sigmoid(self.values[d])
            else:
                self.grad[d] = self.grad[d + 1] @ self.weights[d + 1] * d_sigmoid(self.values[d])

    def standard_bp(self, x, y):
        for _ in range(self.max_iter):
            for xi, yi in zip(x, y):
                # forward propagation
                self.forward_propagation(xi, lambd=0)
                # back propagation
                self.back_propagation(yi)
                # update weight
                for d in range(self.deep):
                    if d == 0:  # input layer to hidden layer
                        self.weights[d] += self.grad[d].reshape(-1, 1) @ xi.reshape(1, -1) * self.eta
                    else:  # the others
                        self.weights[d] += self.grad[d].reshape(-1, 1) @ self.values[d - 1].reshape(1, -1) * self.eta

    def fit(self, x, y):
        x, y = self.preprocessing(x, y)
        self.standard_bp(x, y)
        return self

    def predict(self, x, probability=False):
        x = self.preprocessing(x)[0]
        prob = self.forward_propagation(x)
        if self.label_n == 2:  # binary classification
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
    df = pd.read_csv('hw3_wine.csv', sep='\t')
    col_class = df.pop('# class')
    df.insert(len(df.columns), '# class', col_class)
    col_mean = df.mean().tolist()

    list_target = df['# class'].unique()
    df2 = df[df['# class'].isin([list_target[0]])]
    df1 = df[df['# class'].isin([list_target[1]])]
    df0 = df[df['# class'].isin([list_target[2]])]
    # Split into folds
    k_fold = []
    fold_size2 = int(len(df2) / 10)
    fold_size1 = int(len(df1) / 10)
    fold_size0 = int(len(df0) / 10)
    for k in range(0, 9):
        fold2 = df2.sample(n=fold_size2)
        fold1 = fold2.append(df1.sample(n=fold_size1))
        fold0 = fold1.append(df0.sample(n=fold_size0))
        df2 = df2[~df2.index.isin(fold2.index)]
        df1 = df1[~df1.index.isin(fold1.index)]
        df0 = df0[~df0.index.isin(fold0.index)]
        k_fold.append(fold0)
    k_fold.append(df2.append(df1.append(df0)))

    classLabel_rf_unzip = []
    # Split to train and test dataset
    k_fold_copy = k_fold.copy()
    data_test = k_fold[0]
    del k_fold_copy[0]
    data_train = pd.concat(k_fold_copy).sample(n=len(df) - len(data_test.index), replace=True)
    X_train = MinMaxScaler().fit_transform(data_train.drop('# class', axis=1).values)
    y_train = data_train['# class'].values - 1
    X_test = MinMaxScaler().fit_transform(data_test.drop('# class', axis=1).values)
    y_test = data_test['# class'].values - 1
    J_loop = []
    J_final = []
    for n_sample in range(1, len(y_train)):
        for loop in range(0, 100):
            X_train = MinMaxScaler().fit_transform(data_train.drop('# class', axis=1).values)
            y_train = data_train['# class'].values - 1
            X_train = np.delete(X_train, range(0, n_sample), axis=0)
            y_train = np.delete(y_train, range(0, n_sample), axis=0)
            classifier = BPNNClassifier(feature_n=13, hidden_n=7, deep=2, label_n=3).fit(X_train, y_train)
            [prediction, probability] = classifier.predict(X_test, probability=True)
            J = -np.sum(np.log(probability) * BPNNClassifier.encoder(classifier, y_test)) / len(y_test)
            J_loop.append(J)
            print(loop)
        print(n_sample)
        print('J =', np.mean(J_loop))
        J_final.append(np.mean(J_loop))

    # classifier = BPNNClassifier(feature_n=13, hidden_n=7, deep=2, label_n=3).fit(X_train, y_train)
    # [prediction, probability] = classifier.predict(X_test, probability=True)
    # J = -np.sum(np.log(probability) * BPNNClassifier.encoder(classifier, y_test)) / len(y_test)
    # print('J =', J)
