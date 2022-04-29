import numpy as np
import pandas as pd


def minmax_scale(data):
    mins = data.min(0)
    maxs = data.max(0)
    ranges = maxs - mins
    row = data.shape[0]
    normData = data - np.tile(mins, (row, 1))
    normData = normData / np.tile(ranges, (row, 1))
    return normData


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
                        self.weights[d] += self.grad[d].reshape(-1, 1) @ self.values[d - 1].reshape(1,
                                                                                                    -1) * self.eta * (
                                                   1 - self.lmbda)

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
    X_train = minmax_scale(data_train.drop('# class', axis=1).values)
    y_train = data_train['# class'].values - 1
    X_test = minmax_scale(data_test.drop('# class', axis=1).values)
    y_test = data_test['# class'].values - 1
    J_loop = []
    J_final = []
    for n_sample in range(1, len(y_train), 5):
        for loop in range(0, 100):
            X_train = minmax_scale(data_train.drop('# class', axis=1).values)
            y_train = data_train['# class'].values - 1
            X_train = np.delete(X_train, range(0, n_sample), axis=0)
            y_train = np.delete(y_train, range(0, n_sample), axis=0)
            classifier = BPNNClassifier(in_n=13, hid_l=2, hid_n=2, out_n=3, lmbda=0.05).fit(X_train, y_train)
            [prediction, probability] = classifier.predict(X_test, probability=True)
            J = -np.sum(np.log(probability) * BPNNClassifier.encoder(classifier, y_test)) / len(y_test)
            J_loop.append(J)
            print(loop)
        print(n_sample)
        print('J =', np.mean(J_loop))
        J_final.append(np.mean(J_loop))
