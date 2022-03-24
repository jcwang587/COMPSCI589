import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import TreePlot


def split_data(data_input, split_index, split_category):
    data_output = []
    for data in data_input:
        if data[split_index] == split_category:
            new_data = data[:split_index]
            new_data.extend(data[split_index + 1:])
            data_output.append(new_data)
    return data_output


def calculate_entropy(data_input):
    label_list = [data[-1] for data in data_input]
    entropy = 0
    for key in Counter(label_list).keys():
        probability = float(Counter(label_list)[key]) / len(data_input)
        entropy -= probability * np.log2(probability)
    return entropy


def compare_information_gain(data_input):
    branch_index = 0
    information_gain = []
    for i_branch in range(len(data_input[0]) - 1):
        new_entropy = 0
        branch_data = set([data[i_branch] for data in data_input])
        for category in branch_data:
            new_data = split_data(data_input, i_branch, category)
            probability = len(new_data) / len(data_input)
            new_entropy += probability * calculate_entropy(new_data)
        # Compare the information gain
        information_gain.append(calculate_entropy(data_input) - new_entropy)
        branch_index = information_gain.index(max(information_gain))
    return branch_index


def create_decision_tree(data_input, attribute):
    label_list = [data[-1] for data in data_input]
    # If all instances belong to the same class
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # If there are no more attributes that can be tested
    if len(data_input[0]) == 1:
        return max(label_list, key=label_list.count)
    # Decide the attribute
    branch_index = compare_information_gain(data_input)
    branch = attribute[branch_index]
    decision_tree = {branch: {}}
    branch_data = set([data[branch_index] for data in data_input])
    del (attribute[branch_index])
    for category in branch_data:
        decision_tree[branch][category] = create_decision_tree(
            split_data(data_input, branch_index, category), attribute[:])
    return decision_tree


def predict(tree, attribute_list, test_data):
    label = list(tree.keys())[0]
    dictionary = tree[label]
    value = test_data[attribute_list.index(label)]
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
    # Split the original dataset
    list_target = df['target'].unique()
    df1 = df[df['target'].isin([list_target[0]])]
    df0 = df[df['target'].isin([list_target[1]])]
    # Split into folds
    kfold = []
    fold_size1 = math.ceil(len(df1) / 10)
    fold_size0 = math.ceil(len(df0) / 10)
    while len(df1) > fold_size1:
        fold1 = df1.sample(n=fold_size1)
        fold0 = fold1.append(df0.sample(n=fold_size0))
        df1 = df1[~df1.index.isin(fold1.index)]
        df0 = df0[~df0.index.isin(fold0.index)]
        kfold.append(fold0)
    kfold.append(df1.append(df0))

    while iteration < 1:
        print(iteration)
        try:
            # Shuffle the dataset
            df_sf = shuffle(kfold[iteration])
            y = df_sf[df.columns[16]]
            # Randomly partition the dataset
            data_train, data_test, y_train, y_test = train_test_split(df_sf, y, test_size=0.2)
            data_train

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
    plt.show()
    TreePlot.createPlot(decisionTree)
