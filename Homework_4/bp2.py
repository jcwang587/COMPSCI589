import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNetwork:

    def __init__(self, X, y):
        self.X = (X - X.min()) / (X.max() - X.min())
        self.y = y
        self.H1_size = 256
        self.H2_size = 64
        self.OUTPUT_SIZE = len(np.unique(y))
        self.INPUT_SIZE = X.shape[1]
        self.losses = []

        # Initialize weights
        self.W1 = np.random.randn(self.INPUT_SIZE, self.H1_size)
        self.W2 = np.random.randn(self.H1_size, self.H2_size)
        self.W3 = np.random.randn(self.H2_size, self.OUTPUT_SIZE)

        # Initialize biases
        self.b1 = np.random.random((1, self.H1_size))
        self.b2 = np.random.random((1, self.H2_size))
        self.b3 = np.random.random((1, self.OUTPUT_SIZE))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, x):
        Z1 = x.dot(self.W1) + self.b1  # (N,256) = (N,784)(784,256)(1,256)
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        Z3 = A2.dot(self.W3) + self.b3
        yhat = self.softmax(Z3)

        self.activations = [A1, A2, yhat]

        return yhat

    def backprop(self, x, y, yhat, learning_rate=0.01):

        A1, A2, yhat = self.activations

        # Compute Gradients
        delta3 = yhat - y
        dldw3 = A2.T.dot(delta3)
        dldb3 = delta3.sum(axis=0, keepdims=True)

        delta2 = delta3.dot(self.W3.T) * (A2 * (1 - A2))
        dldw2 = A1.T.dot(delta2)
        dldb2 = delta2.sum(axis=0, keepdims=True)

        delta1 = delta2.dot(self.W2.T) * (A1 * (1 - A1))
        dldw1 = x.T.dot(delta1)
        dldb1 = delta1.sum(axis=0, keepdims=True)

        # Update Weights
        self.W3 -= dldw3 * learning_rate
        self.b3 -= dldb3 * learning_rate

        self.W2 -= dldw2 * learning_rate
        self.b2 -= dldb2 * learning_rate

        self.W1 -= dldw1 * learning_rate
        self.b1 -= dldb1 * learning_rate

    def get_predictions(self, test):
        yhat = self.forward(test)
        preds = np.argmax(yhat, axis=1)
        return preds

    def accuracy(self, preds, true_labels):
        return (preds == true_labels).mean()

    def get_one_hot_vectors(self, labels):
        klasses = len(np.unique(labels))
        vectors = np.zeros((labels.shape[0], klasses))

        for i, l in enumerate(labels):
            vectors[i, int(l)] = 1
        return vectors

    def compute_loss(self, y, yhat):
        # L = -E[y log(yhat)]
        return -np.sum(y * np.log(yhat))

    def train(self, learning_rate=0.01, epochs=10, batch_size=128):
        y_one_hot_vector = self.get_one_hot_vectors(self.y)

        for e in range(epochs):
            size = 0
            while size + batch_size < self.X.shape[0]:
                x_batch = self.X[size: size + batch_size]
                y_batch = self.y[size: size + batch_size]
                size += batch_size

                y_batch_one_hot = self.get_one_hot_vectors(y_batch)
                yhat_batch = self.forward(x_batch)
                self.backprop(x_batch, y_batch_one_hot, yhat_batch, learning_rate)

            yhat = self.forward(self.X)
            self.losses.append(self.compute_loss(y_one_hot_vector, yhat))
            print("Loss at Epoch [{}]: {}".format(e, self.losses[-1]))


dataset = pd.read_csv('hw3_house_votes_84.csv', sep='\t')
dataset = dataset.values
X = dataset[:, :-1]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
nn = NeuralNetwork(X_train, y_train)

nn.train(learning_rate=0.01, epochs=20, batch_size=128)
