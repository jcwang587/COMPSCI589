import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from utils import *
import math


class MultinomialNaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.items = {}
        self.log_priors = {}
        self.word_counts = {}
        self.vocab = set()

    def group_data(self, x, y):
        data = dict()
        for c in self.classes:
            data[c] = x[np.where(y == c)]
        return data

    def fit(self, x, y):
        n = len(x)
        grouped_data = self.group_data(x, y)
        for c, data in grouped_data.items():
            self.items[c] = len(data)
            self.log_priors[c] = math.log(self.items[c] / n)
            self.word_counts[c] = defaultdict(lambda: 0)
            for text in data:
                counts = Counter(text)
                for word, count in counts.items():
                    if word not in self.vocab:
                        self.vocab.add(word)
                    self.word_counts[c][word] += count
        return self

    def laplace_smoothing(self, word, text_class, alpha):
        numerator = self.word_counts[text_class][word] + alpha
        denominator = self.items[text_class] + alpha * len(self.vocab)
        return math.log(numerator / denominator)

    def predict(self, x, alpha):
        result = []
        for text in x:
            class_scores = {c: self.log_priors[c] for c in self.classes}
            words = set(text)
            for word in words:
                if word not in self.vocab:
                    continue
                for c in self.classes:
                    log_w_given_c = self.laplace_smoothing(word, c, alpha)
                    class_scores[c] += log_w_given_c
            result.append(max(class_scores, key=class_scores.get))
        return result


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


def confusion_matrix(y_true, y_pred):
    tp_tn_idx = np.where(y_true == y_pred)[0].tolist()
    tp = [y_pred[i] for i in tp_tn_idx].count(1)
    tp_fp = y_pred.count(1)
    tp_fn = y_true.tolist().count(1)
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = len(y_true) - tp - fp - fn
    matrix = [[tp, fn], [fp, tn]]
    return matrix


if __name__ == "__main__":
    accuracy_list = []
    alpha_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for alpha in alpha_list:
        percent_positive_instance_train = 0.2
        percent_negative_instance_train = 0.2
        percent_positive_instance_test = 0.2
        percent_negative_instance_test = 0.2

        (pos_train, neg_train, vocab) = load_training_set(percent_positive_instance_train, percent_negative_instance_train)
        (pos_test, neg_test) = load_test_set(percent_positive_instance_test, percent_negative_instance_test)

        print("Number of positive training instances:", len(pos_train))
        print("Number of negative training instances:", len(neg_train))
        print("Number of positive test instances:", len(pos_test))
        print("Number of negative test instances:", len(neg_test))
        print("Vocabulary (training set):", len(vocab))

        pos_train_label = [1] * len(pos_train)
        neg_train_label = [0] * len(neg_train)
        train_data = np.array(pos_train + neg_train, dtype=object)
        train_label = np.array(pos_train_label + neg_train_label, dtype=object)

        MNB = MultinomialNaiveBayes(classes=np.unique(train_label)).fit(train_data, train_label)

        pos_test_label = [1] * len(pos_test)
        neg_test_label = [0] * len(neg_test)
        test_data = np.array(pos_test + neg_test, dtype=object)
        test_label = np.array(pos_test_label + neg_test_label, dtype=object)

        predict_label = MNB.predict(test_data, alpha)

        accuracy = accuracy_score(test_label, predict_label)
        precision = precision_score(test_label, predict_label)
        recall = recall_score(test_label, predict_label)
        print(accuracy)
        accuracy_list.append(accuracy)

    plt.semilogx(alpha_list, accuracy_list, '.-', markersize=10, color='#1f77b4')
    plt.xlabel('Value of alpha')
    plt.ylabel('Accuracy on the test set')
    plt.savefig("MNB_Q2.eps", dpi=600, format="eps")
    plt.show()
