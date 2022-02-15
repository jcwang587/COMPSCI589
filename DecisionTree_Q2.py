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
            subData = data[:split_index]
            subData.extend(data[split_index + 1:])
            data_output.append(subData)
    return data_output


def calculate_entropy(data_input):
    numEntries = len(data_input)  # 数据集长度
    classList = [data[-1] for data in data_input]  # 数据集的类别集合
    classCount = {}  # 类别计数
    shannonEnt = 0.0  # 信息嫡
    # 遍历计数
    for value in classList:
        if value not in classCount.keys():
            classCount[value] = 0
        classCount[value] += 1

    # 计算信息嫡
    for key in classCount.keys():
        prob = float(classCount[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


def chooseBestFeature(dataSet):
    numEntries = len(dataSet)  # 数据集数量
    numFeatures = len(dataSet[0]) - 1  # 属性的数量
    baseEnt = calculate_entropy(dataSet)  # 整个数据集的香农嫡
    bestFeature = -1  # 选出的最好属性的index
    maxGain = 0.0  # 记录最大的信息增益
    for i in range(numFeatures):
        newEnt = 0.0  # 划分之后的信息嫡之和
        featList = [data[i] for data in dataSet]  # 获取所有该属性的所有值
        featSet = set(featList)  # 获取该属性不同的值
        for value in featSet:
            subDataSet = split_data(dataSet, i, value)  # 获取属性值相同的数据集
            prob = float(len(subDataSet)) / numEntries
            newEnt += prob * calculate_entropy(subDataSet)
        newGain = baseEnt - newEnt
        if newGain > maxGain:
            maxGain = newGain
            bestFeature = i
    return bestFeature


def create_decision_tree(data_input, attribute):
    classList = [data[-1] for data in data_input]  # 获取数据集中所属类别的数据
    # 检测数据集是否符合同一个分类，相同则返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 假如属性用完了，那么选择当前数据集中数量最多的类别
    if len(data_input[0]) == 1:
        return max(classList, key=classList.count)
    # 选取最好的属性，采取ID3
    bestFeature = chooseBestFeature(data_input)  # 最好属性的index
    bestFeatLabel = attribute[bestFeature]  # 最好属性的名字
    decisionTree = {bestFeatLabel: {}}  # 存储决策树
    featValues = [data[bestFeature] for data in data_input]
    featValuesSet = set(featValues)  # 不同类别的集合
    del (attribute[bestFeature])
    for value in featValuesSet:
        subLabels = attribute[:]  # 针对划分之后的数据集都要有一个新的labels
        decisionTree[bestFeatLabel][value] = create_decision_tree(split_data(data_input, bestFeature, value), subLabels)
    return decisionTree


def predict(tree, labels, testData):
    firstLabel = list(tree.keys())[0]
    firstLabelIndex = labels.index(firstLabel)
    secondDict = tree[firstLabel]
    value = testData[firstLabelIndex]
    classLabel = None
    if type(secondDict[value]).__name__ == "dict":
        classLabel = predict(secondDict[value], labels, testData)
    else:
        classLabel = secondDict[value]
    return classLabel


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
            X = df_sf[df.columns[0:17]]
            y = df_sf[df.columns[16]]

            # Randomly partition the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Normalize the dataset
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
            X_train_data_list = X_train.values.tolist()
            X_test_data_list = X_test.values.tolist()
            X_train_attribute_list = X_train.keys().to_list()

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
