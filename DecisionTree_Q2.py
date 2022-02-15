import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def split_data(data_input, split_index, value):
    data_output = []
    for data in data_input:
        if data[split_index] == value:
            new_data = data[:split_index]
            new_data.extend(data[split_index + 1:])
            data_output.append(new_data)
    return data_output


def calculate_entropy(data_input):
    label_list = [data[-1] for data in data_input]
    classCount = {}  # 类别计数
    entropy = 0
    for value in label_list:
        if value not in classCount.keys():
            classCount[value] = 0
        classCount[value] += 1
    for key in classCount.keys():
        prob = float(classCount[key]) / len(data_input)
        entropy -= prob * math.log(prob, 2)
    return entropy


def compare_information_gain(data_input):
    branch_index = -1  # 选出的最好属性的index
    information_gain = 0.0  # 记录最大的信息增益
    for i in range(len(data_input[0]) - 1):
        newEnt = 0.0  # 划分之后的信息嫡之和
        featList = [data[i] for data in data_input]  # 获取所有该属性的所有值
        featSet = set(featList)  # 获取该属性不同的值
        for value in featSet:
            subDataSet = split_data(data_input, i, value)  # 获取属性值相同的数据集
            prob = float(len(subDataSet)) / len(data_input)
            newEnt += prob * calculate_entropy(subDataSet)
        # Calculate information gain
        newGain = calculate_entropy(data_input) - newEnt
        if newGain > information_gain:
            information_gain = newGain
            branch_index = i
    return branch_index


def create_decision_tree(data_input, attribute):
    classList = [data[-1] for data in data_input]  # 获取数据集中所属类别的数据
    # If all instances belong to the same class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # If there are no more attributes that can be tested
    if len(data_input[0]) == 1:
        return max(classList, key=classList.count)
    # 选取最好的属性，采取ID3
    branch_index = compare_information_gain(data_input)  # 最好属性的index
    branch = attribute[branch_index]  # 最好属性的名字
    decision_tree = {branch: {}}  # 存储决策树
    featValues = [data[branch_index] for data in data_input]
    featValuesSet = set(featValues)  # 不同类别的集合
    del (attribute[branch_index])
    for value in featValuesSet:
        new_label = attribute[:]
        decision_tree[branch][value] = create_decision_tree(split_data(data_input, branch_index, value), new_label)
    return decision_tree


def predict(tree, attribute_list, test_data):
    label = list(tree.keys())[0]
    label_index = attribute_list.index(label)
    dictionary = tree[label]
    value = test_data[label_index]
    if type(dictionary[value]).__name__ == "dict":
        prediction = predict(dictionary[value], attribute_list, test_data)
    else:
        prediction = dictionary[value]
    return prediction


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('house_votes_84.csv')
    iteration = 0
    accuracy = []
    while iteration < 100:
        print(iteration)
        try:
            # Shuffle the dataset
            df_sf = shuffle(df)
            y = df_sf[df.columns[16]]

            # Randomly partition the dataset
            data_train, data_test, y_train, y_test = train_test_split(df_sf, y, test_size=0.2)

            # Normalize the dataset
            X_train_data_list = data_train.values.tolist()
            X_test_data_list = data_test.values.tolist()
            X_train_attribute_list = data_train.keys().to_list()

            # Create decision tree
            X_train_attribute_list_copy = X_train_attribute_list[:]
            decisionTree = create_decision_tree(X_train_data_list, X_train_attribute_list_copy)
            print(decisionTree)

            # Make prediction
            correct = 0
            for index in range(0, len(y_test)):
                classLabel = predict(decisionTree, X_train_attribute_list, X_test_data_list[index])
                if classLabel == y_test.values[index]:
                    correct += 1
            accuracy.append(correct / len(y_test))
            iteration += 1
        except:
            pass
            continue
    std = np.std(accuracy)
    avg = np.mean(accuracy)
    plt.hist(accuracy, weights=np.ones_like(accuracy) / len(accuracy), align="left", rwidth=0.9)
    plt.xlabel('Accuracy')
    plt.ylabel('Accuracy frequency over training data')
    plt.savefig("Figure4.eps", dpi=600, format="eps")
    plt.show()
