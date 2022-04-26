import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dsigmoid(h):
    return h * (1 - h)


class BPNNClassifier:
    def __init__(self, feature_n, hidden_n=10, deep=2, label_n=2, eta=0.1, max_iter=200):
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

    def forward_propagation(self, x):
        self.values.clear()
        value = None
        for d in range(self.deep):
            if d == 0:  # input layer to hidden layer
                value = sigmoid(self.linear_input(d, x))
            elif d == self.deep - 1:  # hidden layer to output layer, use sigmoid
                value = sigmoid(self.linear_input(d, value))
            else:  # the others
                value = sigmoid(self.linear_input(d, value))
            self.values.append(value)
        return value

    def back_propagation(self, y_true):
        for d in range(self.deep - 1, -1, -1):
            if d == self.deep - 1:  # hidden layer to output layer
                self.grad[d] = (y_true - self.values[d]) * dsigmoid(self.values[d])
            else:
                self.grad[d] = self.grad[d + 1] @ self.weights[d + 1] * dsigmoid(self.values[d])

    def standard_bp(self, x, y):
        for _ in range(self.max_iter):
            for xi, yi in zip(x, y):
                # forward propagation
                self.forward_propagation(xi)
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
    kfold = []
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
        kfold.append(fold0)
    kfold.append(df2.append(df1.append(df0)))

    iteration = 0
    while iteration < 1:
        classLabel_rf_unzip = []
        # Split to train and test dataset
        kfold_copy = kfold.copy()
        data_test = kfold[iteration]
        del kfold_copy[iteration]
        data_train = pd.concat(kfold_copy).sample(n=len(df) - len(data_test.index), replace=True)
        X_train = MinMaxScaler().fit_transform(data_train.drop('# class', axis=1).values)
        y_train = data_train['# class'].values - 1
        X_test = MinMaxScaler().fit_transform(data_test.drop('# class', axis=1).values)
        y_test = data_test['# class'].values - 1
        # X = MinMaxScaler().fit_transform(kfold[iteration].iloc[:, :-1].values)

        classifier = BPNNClassifier(feature_n=13, hidden_n=7, deep=2, label_n=3).fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))
        iteration += 1
