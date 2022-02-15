import operator
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# 选取数量最多的类别
def majorityClass(classList):
    classCount = {}  # 类别计数 类别:数量
    # 遍历进行计数
    for value in classList:
        if value not in classCount.keys():
            classCount[value] = 0
        classCount[value] += 1
    # 选择数量最多的类别
    classSort = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return classSort[0][0]


# 数据集进行划分
def splitDataSet(dataSet, index, value):
    subDataSet = []
    for data in dataSet:
        if data[index] == value:
            subData = data[:index]
            subData.extend(data[index + 1:])
            subDataSet.append(subData)
    return subDataSet


# 计算信息嫡
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 数据集长度
    classList = [data[-1] for data in dataSet]  # 数据集的类别集合
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


# 选择最好的属性，ID3
def chooseBestFeature(dataSet):
    numEntries = len(dataSet)  # 数据集数量
    numFeatures = len(dataSet[0]) - 1  # 属性的数量
    baseEnt = calcShannonEnt(dataSet)  # 整个数据集的香农嫡
    bestFeature = -1  # 选出的最好属性的index
    maxGain = 0.0  # 记录最大的信息增益
    for i in range(numFeatures):
        newEnt = 0.0  # 划分之后的信息嫡之和
        featList = [data[i] for data in dataSet]  # 获取所有该属性的所有值
        featSet = set(featList)  # 获取该属性不同的值
        for value in featSet:
            subDataSet = splitDataSet(dataSet, i, value)  # 获取属性值相同的数据集
            prob = float(len(subDataSet)) / numEntries
            newEnt += prob * calcShannonEnt(subDataSet)
        newGain = baseEnt - newEnt
        if newGain > maxGain:
            maxGain = newGain
            bestFeature = i
    return bestFeature


# 判断样本在属性集合上取值是否相同
def propertyIsSame(dataSet):
    numFeatures = len(dataSet[0]) - 1
    for i in range(numFeatures):
        temp = dataSet[0][i]
        for data in dataSet:
            if data[i] != temp:
                return 0
    return 1


# 递归创建决策树
def createTree(dataSet, labels):
    classList = [data[-1] for data in dataSet]  # 获取数据集中所属类别的数据
    # 检测数据集是否符合同一个分类，相同则返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 假如属性用完了，那么选择当前数据集中数量最多的类别
    if len(dataSet[0]) == 1:
        return max(classList, key=classList.count)
    # 选取最好的属性，采取ID3
    bestFeature = chooseBestFeature(dataSet)  # 最好属性的index
    bestFeatLabel = labels[bestFeature]  # 最好属性的名字
    decisionTree = {bestFeatLabel: {}}  # 存储决策树
    featValues = [data[bestFeature] for data in dataSet]
    featValuesSet = set(featValues)  # 不同类别的集合
    del (labels[bestFeature])
    for value in featValuesSet:
        subLabels = labels[:]  # 针对划分之后的数据集都要有一个新的labels
        decisionTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return decisionTree


# 使用决策树进行预测
def classifyDecisionTree(tree, labels, testData):
    firstLabel = list(tree.keys())[0]
    firstLabelIndex = labels.index(firstLabel)
    secondDict = tree[firstLabel]
    value = testData[firstLabelIndex]
    classLabel = None
    if type(secondDict[value]).__name__ == "dict":
        classLabel = classifyDecisionTree(secondDict[value], labels, testData)
    else:
        classLabel = secondDict[value]
    return classLabel


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv('house_votes_84.csv')
    iteration = 0
    accuracy = []
    while iteration < 100:
        print(iteration)
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
        X_train_label_list = X_train.keys().to_list()

        # 复制一份属性标签
        # createTree()操作会影响传入的类别标签
        X_train_label_list_copy = X_train_label_list[:]

        # 创建树
        decisionTree = createTree(X_train_data_list, X_train_label_list_copy)
        print(decisionTree)

        # 进行预测
        correct = 0
        for index in range(0, len(y_train)):
            classLabel = classifyDecisionTree(decisionTree, X_train_label_list, X_train_data_list[index])
            if classLabel == y_train.values[index]:
                correct += 1
        accuracy.append(correct / len(y_train))
        iteration += 1
    std = np.std(accuracy)
    avg = np.mean(accuracy)
    plt.hist(accuracy, weights=np.ones_like(accuracy) / len(accuracy), align="left", rwidth=0.9)
    plt.xlabel('Accuracy')
    plt.ylabel('Accuracy frequency over training data')
    plt.savefig("Figure3.eps", dpi=600, format="eps")
    plt.show()
