# python-Error Back Propagation
# coding=utf-8
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def loss_derivative(output_activations, y):
    return 2 * (output_activations - y)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) * np.tanh(z)


def mean_squared_error(predictY, realY):
    Y = numpy.array(realY)
    return np.sum((predictY - Y) ** 2) / realY.shape[0]


class BP:
    def __init__(self, sizes, activity, activity_derivative, loss_derivative):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((nueron, 1)) for nueron in sizes[1:]]
        self.weights = [np.random.randn(next_layer_nueron, nueron) for nueron, next_layer_nueron in
                        zip(sizes[:-1], sizes[1:])]
        self.activity = activity
        self.activity_derivative = activity_derivative
        self.loss_derivative = loss_derivative

    def predict(self, a):
        re = a.T
        n = len(self.biases) - 1
        for i in range(n):
            b, w = self.biases[i], self.weights[i]
            re = self.activity(np.dot(w, re) + b)
        re = np.dot(self.weights[n], re) + self.biases[n]
        return re.T

    def update_batch(self, batch, learning_rate):
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_temp_b, delta_temp_w = self.update_parameter(x, y)
            temp_w = [w + dw for w, dw in zip(temp_w, delta_temp_w)]
            temp_b = [b + db for b, db in zip(temp_b, delta_temp_b)]
        self.weights = [sw - (learning_rate / len(batch)) * w for sw, w in zip(self.weights, temp_w)]
        self.biases = [sb - (learning_rate / len(batch)) * b for sb, b in zip(self.biases, temp_b)]

    def update_parameter(self, x, y):
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        n = len(self.biases)
        for i in range(n):
            b, w = self.biases[i], self.weights[i]
            z = np.dot(w, activation) + b
            zs.append(z)
            if i != n - 1:
                activation = self.activity(z)
            else:
                activation = z
            activations.append(activation)
        d = self.loss_derivative(activations[-1], y)
        temp_b[-1] = d
        temp_w[-1] = np.dot(d, activations[-2].T)
        for i in range(2, self.num_layers):
            z = zs[-i]
            d = np.dot(self.weights[-i + 1].T, d) * self.activity_derivative(z)
            temp_b[-i] = d
            temp_w[-i] = np.dot(d, activations[-i - 1].T)
        return (temp_b, temp_w)

    def fit(self, train_data, epochs, batch_size, learning_rate, validation_data=None):
        n = len(train_data)
        for j in range(epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if (validation_data != None):
                val_pre = self.predict(validation_data[0])
                print("Epoch", j + 1, '/', epochs,
                      '  val loss:%12.12f' % mean_squared_error(val_pre, validation_data[1]))


def load_data(step):
    x = np.array([numpy.mgrid[-5: 5: 10 / step]]).T
    y = numpy.sin(5 * numpy.pi * x / 4) + 8
    return x, y


if __name__ == "__main__":
    numpy.random.seed(7)

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
    k_fold_copy = k_fold.copy()
    data_test = k_fold[0]
    del k_fold_copy[0]
    data_train = pd.concat(k_fold_copy).sample(n=len(df) - len(data_test.index), replace=True)
    X_train = np.array(MinMaxScaler().fit_transform(data_train.drop('# class', axis=1).values))
    y_train = np.array(data_train['# class'].values - 1)
    X_test = np.array(MinMaxScaler().fit_transform(data_test.drop('# class', axis=1).values))
    y_test = np.array(data_test['# class'].values - 1)

    beta = 1e-3
    layer = [13, 10, 3]
    data = [(np.array([x_value]), np.array([y_value])) for x_value, y_value in zip(X_train, y_train)]
    model = BP(layer, tanh, tanh_derivative, loss_derivative)
    model.fit(train_data=data, epochs=100, batch_size=20, learning_rate=beta, validation_data=(X_test, y_train))
    predict = model.predict(X_test)
