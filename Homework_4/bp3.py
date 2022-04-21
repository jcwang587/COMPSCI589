# -*- coding: utf-8 -*-
import numpy as np


# 代价函数
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]  # theta的中长度
    # 还原theta1和theta2
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels, hidden_layer_size + 1)

    # np.savetxt("Theta1.csv",Theta1,delimiter=',')

    m = X.shape[0]
    class_y = np.zeros((m, num_labels))  # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]
    # 正则化向theta^2
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1)))),
                  np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1))))

    '''正向传播,每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    h = sigmoid(z3)
    '''代价'''
    J = -(np.dot(np.transpose(class_y.reshape(-1, 1)), np.log(h.reshape(-1, 1))) + np.dot(
        np.transpose(1 - class_y.reshape(-1, 1)), np.log(1 - h.reshape(-1, 1))) - Lambda * term / 2) / m
    # temp1 = (h.reshape(-1,1)-class_y.reshape(-1,1))
    # temp2 = (temp1**2).sum()
    # J = 1/(2*m)*temp2
    return np.ravel(J)


# 梯度
def nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size,
                                                                             input_layer_size + 1).copy()
    # 这里使用copy函数，否则下面修改Theta的值，nn_params也会一起修改
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels,
                                                                                  hidden_layer_size + 1).copy()
    m = X.shape[0]
    class_y = np.zeros((m, num_labels))  # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]

    Theta1_grad = np.zeros((Theta1.shape))  # 第一层到第二层的权重
    Theta2_grad = np.zeros((Theta2.shape))  # 第二层到第三层的权重

    '''正向传播，每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    h = sigmoid(z3)

    '''反向传播，delta为误差，'''
    delta3 = np.zeros((m, num_labels))
    delta2 = np.zeros((m, hidden_layer_size))
    for i in range(m):
        # delta3[i,:] = (h[i,:]-class_y[i,:])*sigmoidGradient(z3[i,:])  # 均方误差的误差率
        delta3[i, :] = h[i, :] - class_y[i, :]  # 交叉熵误差率
        Theta2_grad = Theta2_grad + np.dot(np.transpose(delta3[i, :].reshape(1, -1)), a2[i, :].reshape(1, -1))
        delta2[i, :] = np.dot(delta3[i, :].reshape(1, -1), Theta2_x) * sigmoidGradient(z2[i, :])
        Theta1_grad = Theta1_grad + np.dot(np.transpose(delta2[i, :].reshape(1, -1)), a1[i, :].reshape(1, -1))

    Theta1[:, 0] = 0
    Theta2[:, 0] = 0
    '''梯度'''
    grad = (np.vstack((Theta1_grad.reshape(-1, 1), Theta2_grad.reshape(-1, 1))) + Lambda * np.vstack(
        (Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))) / m
    return np.ravel(grad)


# S型函数
def sigmoid(z):
    h = np.zeros((len(z), 1))  # 初始化，与z的长度一致

    h = 1.0 / (1.0 + np.exp(-z))
    return h


# S型函数导数
def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# 初始化调试的theta权重
def debugInitializeWeights(fan_in, fan_out):
    W = np.zeros((fan_out, fan_in + 1))
    x = np.arange(1, fan_out * (fan_in + 1) + 1)
    W = np.sin(x).reshape(W.shape) / 10
    return W


if __name__ == "__main__":
    Lambda = 0
    '''构造一个小型的神经网络验证，因为数值法计算梯度很浪费时间，而且验证正确后之后就不再需要验证了'''
    input_layer_size = 1
    hidden_layer_size = 2
    num_labels = 1
    m = 5
    initial_Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = debugInitializeWeights(hidden_layer_size, num_labels)
    X = debugInitializeWeights(input_layer_size - 1, m)
    y = np.transpose(np.mod(np.arange(1, m + 1), num_labels))  # 初始化y

    y = y.reshape(-1, 1)
    nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))  # 展开theta
    '''BP求出梯度'''
    grad = nnGradient(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, Lambda)
    '''使用数值法计算梯度'''
    num_grad = np.zeros((nn_params.shape[0]))
    step = np.zeros((nn_params.shape[0]))
    e = 1e-4
    for i in range(nn_params.shape[0]):
        step[i] = e
        loss1 = nnCostFunction(nn_params - step.reshape(-1, 1), input_layer_size, hidden_layer_size,
                               num_labels, X, y,
                               Lambda)
        loss2 = nnCostFunction(nn_params + step.reshape(-1, 1), input_layer_size, hidden_layer_size,
                               num_labels, X, y,
                               Lambda)
        num_grad[i] = (loss2 - loss1) / (2 * e)
        step[i] = 0
    # 显示两列比较
    res = np.hstack((num_grad.reshape(-1, 1), grad.reshape(-1, 1)))
    print("检查梯度的结果，第一列为数值法计算得到的，第二列为BP得到的:")
    print(res)
