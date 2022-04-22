import numpy as np


def nn_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size,
                                                                             input_layer_size + 1).copy()
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels,
                                                                                  hidden_layer_size + 1).copy()
    m = X.shape[0]

    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    a3 = sigmoid(z3)
    h = a3
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1)))),
                  np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1))))
    J = -(np.dot(np.transpose(y.reshape(-1, 1)), np.log(h.reshape(-1, 1))) + np.dot(
        np.transpose(1 - y.reshape(-1, 1)), np.log(1 - h.reshape(-1, 1))) - Lambda * term / 2) / m

    print('Computing the error/cost, J, of the network')
    print('Processing training instance 1')
    print('Forward propagating the input [%6.5f]' % X[0, 0])
    print('a1: [%6.5f %6.5f]' % (a1[0, 0], a1[0, 1]))
    print('z2: [%6.5f %6.5f]' % (z2[0, 0], z2[0, 1]))
    print('a2: [%6.5f %6.5f %6.5f]' % (a2[0, 0], a2[0, 1], a2[0, 2]))
    print('z3: [%6.5f]' % z3[0, 0])
    print('a3: [%6.5f]' % a3[0, 0])
    print('f(x): [%6.5f]' % h[0, 0])
    print('Predicted output for instance 1: [%6.5f]' % h[0, 0])
    print('Expected output for instance 1: [%6.5f]' % y[0, 0])
    print('Cost, J, associated with instance 1: [%6.5f]' % J[0, 0])

    print('Processing training instance 2')
    print('Forward propagating the input [%6.5f]' % X[1, 0])
    print('a1: [%6.5f %6.5f]' % (a1[1, 0], a1[1, 1]))
    print('z2: [%6.5f %6.5f]' % (z2[1, 0], z2[1, 1]))
    print('a2: [%6.5f %6.5f %6.5f]' % (a2[1, 0], a2[1, 1], a2[1, 2]))
    print('z3: [%6.5f]' % z3[1, 0])
    print('a3: [%6.5f]' % a3[1, 0])
    print('f(x): [%6.5f]' % h[1, 0])
    print('Predicted output for instance 2: [%6.5f]' % h[1, 0])
    print('Expected output for instance 2: [%6.5f]' % y[1, 0])
    print('Cost, J, associated with instance 2: [%6.5f]' % J[0, 0])
    print('Final (regularized) cost, J, based on the complete training set: [%6.5f]' % J)

    delta3 = np.zeros((m, num_labels))
    delta2 = np.zeros((m, hidden_layer_size))
    Theta1_grad = []
    Theta2_grad = []
    for i in range(m):
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)
        delta3[i, :] = h[i, :] - y[i, :]
        Theta2_grad = Theta2_grad + np.dot(np.transpose(delta3[i, :].reshape(1, -1)), a2[i, :].reshape(1, -1))
        delta2[i, :] = np.dot(delta3[i, :].reshape(1, -1), Theta2_x) * sigmoid_gradient(z2[i, :])
        Theta1_grad = Theta1_grad + np.dot(np.transpose(delta2[i, :].reshape(1, -1)), a1[i, :].reshape(1, -1))
        print('Theta2_grad', Theta2_grad)
        print('Theta1_grad', Theta1_grad)
    print('delta3', delta3)
    print('delta2', delta2)

    Theta1[:, 0] = 0
    Theta2[:, 0] = 0
    gradient = (np.vstack((Theta1_grad.reshape(-1, 1), Theta2_grad.reshape(-1, 1))) +
                Lambda * np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))) / m
    return np.ravel(gradient)


def sigmoid(z):
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


if __name__ == "__main__":
    Lambda = 0
    input_layer_size = 1
    hidden_layer_size = 2
    num_labels = 1
    initial_Theta1 = np.array([[0.4, 0.1], [0.3, 0.2]])
    initial_Theta2 = np.array([[0.7, 0.5, 0.6]])
    X = np.array([[0.13], [0.42]])
    y = np.array([[0.9], [0.23]])
    y = y.reshape(-1, 1)
    nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
    grad = nn_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
