import numpy as np


def nn_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):
    length1 = nn_params[0].shape[0]
    length2 = nn_params[1].shape[0]
    Theta1 = nn_params[0][0:hidden_layer_size[0] * (input_layer_size + 1)].reshape(hidden_layer_size[0],
                                                                                   input_layer_size + 1).copy()
    Theta2 = nn_params[0][hidden_layer_size[0] * (input_layer_size + 1):length1].reshape(hidden_layer_size[1],
                                                                                         hidden_layer_size[
                                                                                             0] + 1).copy()
    Theta3 = nn_params[1][hidden_layer_size[1] * (hidden_layer_size[0] + 1):length2].reshape(num_labels,
                                                                                             hidden_layer_size[
                                                                                                 1] + 1).copy()
    m = X.shape[0]
    n = X.shape[1]

    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]
    Theta3_colCount = Theta3.shape[1]
    Theta3_x = Theta3[:, 1:Theta3_colCount]

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    a3 = sigmoid(z3)
    a3 = np.hstack((np.ones((m, 1)), a3))
    z4 = np.dot(a3, np.transpose(Theta3))
    a4 = sigmoid(z4)
    h = a4

    J1 = - np.dot(np.transpose(y[:n].reshape(-1, 1)), np.log(h[0].reshape(-1, 1))) - np.dot(
        np.transpose(1 - y[:n].reshape(-1, 1)), np.log(1 - h[0].reshape(-1, 1)))
    J2 = - np.dot(np.transpose(y[n:].reshape(-1, 1)), np.log(h[1].reshape(-1, 1))) - np.dot(
        np.transpose(1 - y[n:].reshape(-1, 1)), np.log(1 - h[1].reshape(-1, 1)))
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1), Theta3_x.reshape(-1, 1)))),
                  np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1), Theta3_x.reshape(-1, 1))))
    J = ((J1 + J2) + lambd * term / 2) / m

    print('Computing the error/cost, J, of the network')
    print('Processing training instance 1')
    print('Forward propagating the input [%6.5f %6.5f]' % (X[0, 0], X[0, 1]))
    print('a1: [%6.5f %6.5f %6.5f]' % (a1[0, 0], a1[0, 1], a1[0, 2]))
    print('z2: [%6.5f %6.5f %6.5f %6.5f]' % (z2[0, 0], z2[0, 1], z2[0, 2], z2[0, 3]))
    print('a2: [%6.5f %6.5f %6.5f %6.5f %6.5f]' % (a2[0, 0], a2[0, 1], a2[0, 2], a2[0, 3], a2[0, 4]))
    print('z3: [%6.5f %6.5f %6.5f]' % (z3[0, 0], z3[0, 1], z3[0, 2]))
    print('a3: [%6.5f %6.5f %6.5f %6.5f]' % (a3[0, 0], a3[0, 1], a3[0, 2], a3[0, 3]))
    print('z4: [%6.5f %6.5f]' % (z4[0, 0], z4[0, 1]))
    print('a4: [%6.5f %6.5f]' % (a4[0, 0], a4[0, 1]))
    print('f(x): [%6.5f %6.5f]' % (h[0, 0], h[0, 1]))
    print('Predicted output for instance 1: [%6.5f %6.5f]' % (h[0, 0], h[0, 1]))
    print('Expected output for instance 1: [%6.5f  %6.5f]' % (y[0], y[1]))
    print('Cost, J, associated with instance 1: %4.3f' % J1)

    print('Processing training instance 2')
    print('Forward propagating the input [%6.5f %6.5f]' % (X[1, 0], X[1, 1]))
    print('a1: [%6.5f %6.5f %6.5f]' % (a1[1, 0], a1[1, 1], a1[1, 2]))
    print('z2: [%6.5f %6.5f %6.5f %6.5f]' % (z2[1, 0], z2[1, 1], z2[1, 2], z2[1, 3]))
    print('a2: [%6.5f %6.5f %6.5f %6.5f %6.5f]' % (a2[1, 0], a2[1, 1], a2[1, 2], a2[1, 3], a2[1, 4]))
    print('z3: [%6.5f %6.5f %6.5f]' % (z3[1, 0], z3[1, 1], z3[1, 2]))
    print('a3: [%6.5f %6.5f %6.5f %6.5f]' % (a3[1, 0], a3[1, 1], a3[1, 2], a3[1, 3]))
    print('z4: [%6.5f %6.5f]' % (z4[1, 0], z4[1, 1]))
    print('a4: [%6.5f %6.5f]' % (a4[1, 0], a4[1, 1]))
    print('f(x): [%6.5f %6.5f]' % (h[1, 0], h[1, 1]))
    print('Predicted output for instance 2: [%6.5f %6.5f]' % (h[1, 0], h[1, 1]))
    print('Expected output for instance 2: [%6.5f  %6.5f]' % (y[2], y[3]))
    print('Cost, J, associated with instance 2: %4.3f' % J2)
    print('Final (regularized) cost, J, based on the complete training set: %6.5f' % J)
    print('--------------------------------------------')
    delta4 = np.zeros((m, num_labels))
    delta3 = np.zeros((m, hidden_layer_size[1]))
    delta2 = np.zeros((m, hidden_layer_size[0]))
    Theta1_grad_reg = np.zeros(Theta1.shape)
    Theta2_grad_reg = np.zeros(Theta2.shape)
    Theta3_grad_reg = np.zeros(Theta3.shape)
    print('Running backpropagation')
    for i in range(m):
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)
        Theta3_grad = np.zeros(Theta3.shape)

        delta4[i, :] = h[i, :] - y.reshape(2, 2)[i, :]
        Theta3_grad = Theta3_grad + np.dot(np.transpose(delta4[i, :].reshape(1, -1)), a3[i, :].reshape(1, -1))
        delta3[i, :] = np.dot(delta4[i, :].reshape(1, -1), Theta3_x) * sigmoid_gradient(z3[i, :])
        Theta2_grad = Theta2_grad + np.dot(np.transpose(delta3[i, :].reshape(1, -1)), a2[i, :].reshape(1, -1))
        delta2[i, :] = np.dot(delta3[i, :].reshape(1, -1), Theta2_x) * sigmoid_gradient(z2[i, :])
        Theta1_grad = Theta1_grad + np.dot(np.transpose(delta2[i, :].reshape(1, -1)), a1[i, :].reshape(1, -1))
        Theta1_grad_reg += Theta1_grad
        Theta2_grad_reg += Theta2_grad
        Theta3_grad_reg += Theta3_grad
        print('Computing gradients based on training instance %1.0f' % (i + 1))
        print('delta4: [%6.5f %6.5f]' % (delta4[i, 0], delta4[i, 1]))
        print('delta3: [%6.5f %6.5f %6.5f]' % (delta3[i, 0], delta3[i, 1], delta3[i, 2]))
        print('delta2: [%6.5f %6.5f %6.5f %6.5f]' % (delta2[i, 0], delta2[i, 1], delta2[i, 2], delta2[i, 3]))
        print('Gradient for Theta3 based on training instance %1.0f' % (i + 1))
        print('%6.5f %6.5f %6.5f %6.5f' % (Theta3_grad[0, 0], Theta3_grad[0, 1], Theta3_grad[0, 2], Theta3_grad[0, 3]))
        print('%6.5f %6.5f %6.5f %6.5f' % (Theta3_grad[1, 0], Theta3_grad[1, 1], Theta3_grad[1, 2], Theta3_grad[1, 3]))
        print('Gradients of Theta2 based on training instance %1.0f:' % (i + 1))
        print('%6.5f %6.5f %6.5f %6.5f %6.5f' % (Theta2_grad[0, 0], Theta2_grad[0, 1], Theta2_grad[0, 2],
                                                 Theta2_grad[0, 3], Theta2_grad[0, 4]))
        print('%6.5f %6.5f %6.5f %6.5f %6.5f' % (Theta2_grad[1, 0], Theta2_grad[1, 1], Theta2_grad[1, 2],
                                                 Theta2_grad[1, 3], Theta2_grad[1, 4]))
        print('%6.5f %6.5f %6.5f %6.5f %6.5f' % (Theta2_grad[2, 0], Theta2_grad[2, 1], Theta2_grad[2, 2],
                                                 Theta2_grad[2, 3], Theta2_grad[2, 4]))
        print('Gradients of Theta1 based on training instance %1.0f:' % (i + 1))
        print('%6.5f %6.5f %6.5f' % (Theta1_grad[0, 0], Theta1_grad[0, 1], Theta1_grad[0, 2]))
        print('%6.5f %6.5f %6.5f' % (Theta1_grad[1, 0], Theta1_grad[1, 1], Theta1_grad[1, 2]))
        print('%6.5f %6.5f %6.5f' % (Theta1_grad[2, 0], Theta1_grad[2, 1], Theta1_grad[2, 2]))
        print('%6.5f %6.5f %6.5f' % (Theta1_grad[3, 0], Theta1_grad[3, 1], Theta1_grad[3, 2]))

    Theta1_grad_reg = Theta1_grad_reg / m
    Theta2_grad_reg = Theta2_grad_reg / m
    Theta3_grad_reg = Theta3_grad_reg / m

    print('The entire training set has been processes. Computing the average (regularized) gradients:')
    print('Final regularized gradients of Theta1:')
    print('%6.5f %6.5f %6.5f' % (Theta1_grad_reg[0, 0], Theta1_grad_reg[0, 1], Theta1_grad_reg[0, 2]))
    print('%6.5f %6.5f %6.5f' % (Theta1_grad_reg[1, 0], Theta1_grad_reg[1, 1], Theta1_grad_reg[1, 2]))
    print('%6.5f %6.5f %6.5f' % (Theta1_grad_reg[2, 0], Theta1_grad_reg[2, 1], Theta1_grad_reg[2, 2]))
    print('%6.5f %6.5f %6.5f' % (Theta1_grad_reg[3, 0], Theta1_grad_reg[3, 1], Theta1_grad_reg[3, 2]))
    print('Final regularized gradients of Theta2:')
    print('%6.5f %6.5f %6.5f %6.5f %6.5f' % (Theta2_grad_reg[0, 0], Theta2_grad_reg[0, 1], Theta2_grad_reg[0, 2],
                                             Theta2_grad_reg[0, 3], Theta2_grad_reg[0, 4]))
    print('Final regularized gradients of Theta3:')
    print('%6.5f %6.5f %6.5f %6.5f' % (Theta3_grad_reg[0, 0], Theta3_grad_reg[0, 1], Theta3_grad_reg[0, 2],
                                       Theta3_grad_reg[0, 3]))
    grad = np.vstack((Theta1_grad_reg.reshape(-1, 1), Theta2_grad_reg.reshape(-1, 1)))
    return np.ravel(grad)


def sigmoid(z):
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


if __name__ == "__main__":
    lambd = 0.25
    input_layer_size = 2
    hidden_layer_size = [4, 3]
    num_labels = 2
    initial_Theta1 = np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]])
    initial_Theta2 = np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89],
                               [0.03, 0.56, 0.8, 0.69, 0.09]])
    initial_Theta3 = np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])
    X = np.array([[0.32, 0.68], [0.83, 0.02]])
    y = np.array([[0.75, 0.98], [0.75, 0.28]])
    y = y.reshape(-1, 1)
    nn_params = [np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1))),
                 np.vstack((initial_Theta2.reshape(-1, 1), initial_Theta3.reshape(-1, 1)))]
    grad = nn_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
