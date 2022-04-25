import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


class BPNNClassifier:
    def __init__(self, feature_n, hidden_n=10, deep=2, label_n=2, eta=0.1, max_iter=200, activate_func="tanh"):
        self.feature_n = feature_n
        self.hidden_n = hidden_n
        self.deep = deep
        self.label_n = label_n
        self.eta = eta
        self.max_iter = max_iter
        self.weights = []
        self.gradients = list(range(deep))  # save the gradient of every neuron
        self.values = []  # save the activated value of every neuron
        activate_funcs = \
            {"tanh": (self.tanh, self.dtanh), "sigmoid": (self.sigmoid, self.dsigmoid)}
        self.activate_func, self.dactivate_func = activate_funcs[activate_func]
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

    def linear_input(self, deep, X):
        weight = self.weights[deep]
        return X @ weight.T

    def activation(self, z, func):
        return func(z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def dsigmoid(self, h):
        return h * (1 - h)

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def dtanh(self, h):
        return 1 - h ** 2

    def preproccessing(self, X=None, y=None):
        X_y = []
        if isinstance(X, np.ndarray):
            X0 = np.array([[1] for i in range(X.shape[0])])
            X = np.hstack([X0, X])
            X_y.append(X)
        if isinstance(y, np.ndarray):
            y = self.encoder(y)
            X_y.append(y)
        return tuple(X_y)

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

    def forward_propagation(self, X):
        self.values.clear()
        value = None
        for d in range(self.deep):
            if d == 0:  # input layer to hidden layer
                value = self.activation(self.linear_input(d, X), self.activate_func)
            elif d == self.deep - 1:  # hidden layer to output layer, use sigmoid
                value = self.activation(self.linear_input(d, value), self.sigmoid)
            else:  # the others
                value = self.activation(self.linear_input(d, value), self.activate_func)
            self.values.append(value)
        return value

    def back_propagation(self, y_true):
        for d in range(self.deep - 1, -1, -1):
            if d == self.deep - 1:  # hidden layer to output layer
                self.gradients[d] = (y_true - self.values[d]) * self.dsigmoid(self.values[d])
            else:
                self.gradients[d] = self.gradients[d + 1] @ self.weights[d + 1] * self.dactivate_func(self.values[d])

    def standard_BP(self, X, y):
        for l in range(self.max_iter):
            for Xi, yi in zip(X, y):
                # forward propagation
                self.forward_propagation(Xi)
                # back propagation
                self.back_propagation(yi)
                # update weight
                for d in range(self.deep):
                    if d == 0:  # input layer to hidden layer
                        self.weights[d] += self.gradients[d].reshape(-1, 1) @ Xi.reshape(1, -1) * self.eta
                    else:  # the others
                        self.weights[d] += self.gradients[d].reshape(-1, 1) @ self.values[d - 1].reshape(1,
                                                                                                         -1) * self.eta

    def fit(self, X, y):
        X, y = self.preproccessing(X, y)
        self.standard_BP(X, y)
        return self

    def predict(self, X, probability=False):
        X = self.preproccessing(X)[0]
        prob = self.forward_propagation(X)
        y = None
        if self.label_n == 2:  # binary classification
            y = np.where(prob >= 0.5, 1, 0)
        else:  # mutiply classification
            y = np.zeros(prob.shape)
            for yi, i in zip(y, np.argmax(prob, axis=1)):
                yi[i] = 1
        y = self.preproccessing(y=y)[0]
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
