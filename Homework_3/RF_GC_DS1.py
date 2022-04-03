import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def accuracy_score(y_true, y_pred):
    score = y_true == y_pred
    return np.average(score)


def precision_score(y_true, y_pred):
    tp_tn_idx = np.where(y_true == y_pred)[0].tolist()
    tp = [y_pred[i] for i in tp_tn_idx].count(1)
    tp_fp = y_pred.count(1)
    return tp / tp_fp


def recall_score(y_true, y_pred):
    tp_tn_idx = np.where(y_true == y_pred)[0].tolist()
    tp = [y_pred[i] for i in tp_tn_idx].count(1)
    tp_fn = y_true.tolist().count(1)
    return tp / tp_fn


def f1_score(precision_value, recall_value):
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def find_max_layer(tree):
    layer = 0
    for key in tree.keys():
        if type(tree[key]).__name__ == "dict":
            layer = max(layer, find_max_layer(tree[key]) + 1)
    return layer


def split_data(data_input, split_index, split_category):
    data_output = []
    for data in data_input:
        if data[split_index] == split_category:
            new_data = data[:split_index]
            new_data.extend(data[split_index + 1:])
            data_output.append(new_data)
    return data_output


def calculate_gini(data_input):
    label_list = [data[-1] for data in data_input]
    gini = 0
    for key in Counter(label_list).keys():
        probability = float(Counter(label_list)[key]) / len(data_input)
        gini += probability ** 2
    return 1 - gini


def compare_information_gain(data_input):
    branch_index = 0
    information_gain = []
    for i_branch in range(len(data_input[0]) - 1):
        new_entropy = 0
        branch_data = set([data[i_branch] for data in data_input])
        for category in branch_data:
            new_data = split_data(data_input, i_branch, category)
            probability = len(new_data) / len(data_input)
            new_entropy += probability * calculate_gini(new_data)
        # Compare the information gain
        information_gain.append(calculate_gini(data_input) - new_entropy)
        branch_index = information_gain.index(max(information_gain))
    return branch_index


def create_decision_tree(data_input, attribute, sample_attribute_number):
    label_list = [data[-1] for data in data_input]
    # If all instances belong to the same class
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # If there are no more attributes that can be tested
    if len(data_input[0]) == 1:
        return max(label_list, key=label_list.count)
    # Randomly select m attribute for sampling
    jdx_range = random.sample(range(0, len(attribute) - 1), sample_attribute_number)
    data_input_sample_unzip = [[kdx[jdx] for kdx in data_input] for jdx in jdx_range] + \
                              [[kdx[-1] for kdx in data_input]]
    data_input_sample = list(map(list, zip(*data_input_sample_unzip)))
    # Decide the attribute to split
    branch_index = jdx_range[compare_information_gain(data_input_sample)]
    branch = attribute[branch_index]
    decision_tree = {branch: {}}
    branch_data = set([data[branch_index] for data in data_input])
    del (attribute[branch_index])
    # Check stopping criteria of maximal depth
    if find_max_layer(decision_tree) > 10:
        return decision_tree
    for category in branch_data:
        decision_tree[branch][category] = create_decision_tree(
            split_data(data_input, branch_index, category), attribute[:], sample_attribute_number)
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
    df = pd.read_csv('hw3_wine.csv', sep='\t')
    col_class = df.pop('# class')
    df.insert(len(df.columns), '# class', col_class)
    col_mean = df.mean().tolist()
    for idx in range(0, len(df.columns) - 1):
        df.loc[df[df.keys()[idx]] <= col_mean[idx], df.keys()[idx]] = 0
        df.loc[df[df.keys()[idx]] > col_mean[idx], df.keys()[idx]] = 1
    # Split the original dataset
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

    # Change the number of trees
    ntree_list = [1, 5, 10, 20, 30, 40, 50]
    n_accuracy = []
    n_precision = []
    n_recall = []
    n_f1 = []
    for ntree in ntree_list:
        iteration = 0
        accuracy = []
        precision = []
        recall = []
        f1 = []
        while iteration < 10:
            itree = 0
            classLabel_rf_unzip = []
            while itree < ntree:
                try:
                    # Split to train and test dataset
                    kfold_copy = kfold[:]
                    data_test = kfold[iteration]
                    del kfold_copy[iteration]
                    data_train = pd.concat(kfold_copy).sample(n=len(df) - len(data_test.index), replace=True)
                    y_test = data_test[data_test.columns[-1]]
                    y_train = data_train[data_train.columns[-1]]
                    # Convert to list format
                    X_train_data_list = data_train.values.tolist()
                    X_test_data_list = data_test.values.tolist()
                    X_train_attribute_list = data_train.keys().to_list()
                    # Create decision tree
                    X_train_attribute_list_copy = X_train_attribute_list[:]
                    m = math.ceil((len(X_train_attribute_list_copy) - 1) ** 0.5)
                    decisionTree = create_decision_tree(X_train_data_list, X_train_attribute_list_copy, m)
                    # Make predictions
                    classLabel_list = []
                    for index in range(0, len(y_test)):
                        classLabel = predict(decisionTree, X_train_attribute_list, X_test_data_list[index])
                        classLabel_list.append(classLabel)
                    classLabel_rf_unzip.append(classLabel_list)
                    itree += 1
                    print(decisionTree)
                    print("ntree: ", itree)
                except:
                    pass
                    continue
            classLabel_rf = list(map(list, zip(*classLabel_rf_unzip)))
            final_prediction = [max(idx, key=idx.count) for idx in classLabel_rf]
            final_prediction = [int(i_prediction) for i_prediction in final_prediction]
            final_true = y_test.values.tolist()
            # Calculate metrics
            final_prediction_1 = [0 if i in [2, 3] else i for i in final_prediction]
            final_true_1 = np.array([0 if i in [2, 3] else i for i in final_true])
            accuracy1 = accuracy_score(final_true_1, final_prediction_1)
            precision1 = precision_score(final_true_1, final_prediction_1)
            recall1 = recall_score(final_true_1, final_prediction_1)
            f11 = 2 * (precision1 * recall1) / (precision1 + recall1)

            final_prediction_2 = [0 if i in [1, 3] else i for i in final_prediction]
            final_prediction_2 = [1 if i == 2 else i for i in final_prediction_2]
            final_true_2 = [0 if i in [1, 3] else i for i in final_true]
            final_true_2 = np.array([1 if i == 2 else i for i in final_true_2])
            accuracy2 = accuracy_score(final_true_2, final_prediction_2)
            precision2 = precision_score(final_true_2, final_prediction_2)
            recall2 = recall_score(final_true_2, final_prediction_2)
            f12 = 2 * (precision2 * recall2) / (precision2 + recall2)

            final_prediction_3 = [0 if i in [1, 2] else i for i in final_prediction]
            final_prediction_3 = [1 if i == 3 else i for i in final_prediction_3]
            final_true_3 = [0 if i in [1, 2] else i for i in final_true]
            final_true_3 = np.array([1 if i == 3 else i for i in final_true_3])
            accuracy3 = accuracy_score(final_true_3, final_prediction_3)
            precision3 = precision_score(final_true_3, final_prediction_3)
            recall3 = recall_score(final_true_3, final_prediction_3)
            f13 = 2 * (precision3 * recall3) / (precision3 + recall3)

            accuracy.append(np.mean([accuracy1, accuracy2, accuracy3]))
            precision.append(np.mean([precision1, precision2, precision3]))
            recall.append(np.mean([recall1, recall2, recall3]))
            f1.append(np.mean([f11, f12, f13]))
            iteration += 1
            print("iteration: ", iteration)
        n_accuracy.append(np.mean(accuracy))
        n_precision.append(np.mean(precision))
        n_recall.append(np.mean(recall))
        n_f1.append(np.mean(f1))

    # Plot the metrics
    plt.plot(ntree_list, n_accuracy, '.-', markersize=10, color='#1f77b4', label='Accuracy')
    plt.xlabel('Value of ntree')
    plt.ylabel('Accuracy of the random forest')
    plt.title('The Wine Dataset')
    plt.savefig("FigureDS1_GC_Accuracy.eps", dpi=600, format="eps")
    plt.show()
    plt.plot(ntree_list, n_precision, '.-', markersize=10, color='#ff7f0e', label='Precision')
    plt.xlabel('Value of ntree')
    plt.ylabel('Precision of the random forest')
    plt.title('The Wine Dataset')
    plt.savefig("FigureDS1_GC_Precision.eps", dpi=600, format="eps")
    plt.show()
    plt.plot(ntree_list, n_recall, '.-', markersize=10, color='#2ca02c', label='Recall')
    plt.xlabel('Value of ntree')
    plt.ylabel('Recall of the random forest')
    plt.title('The Wine Dataset')
    plt.savefig("FigureDS1_GC_Recall.eps", dpi=600, format="eps")
    plt.show()
    plt.plot(ntree_list, n_f1, '.-', markersize=10, color='#d62728', label='F1')
    plt.xlabel('Value of ntree')
    plt.ylabel('F1 of the random forest')
    plt.title('The Wine Dataset')
    plt.savefig("FigureDS1_GC_F1.eps", dpi=600, format="eps")
    plt.show()
