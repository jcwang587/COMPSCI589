import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import *


def words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for words in input_set:
        if words in vocab_list:
            return_vec[vocab_list.index(words)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % words)
    return return_vec


class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

        self._dic_class_prior = {}
        self._cd_prob = {}

    def fit(self, x, y):
        # calculate class prior probabilities: P(y=ck)
        self._cal_y_prob(y)

        # calculate Conditional Probability: P( xj | y=ck )
        self._cal_x_prob(x, y)

    def _cal_y_prob(self, y):
        """
        calculate class prior probability
        like: {class_1: prob_1, class_2:prob_2, ...}
        for example two class 1, 2 with probability 0.4 and 0.6
        {1: 0.4, 2: 0.6}
        """
        sample_num = len(y) * 1.0
        unique_class, class_count = np.unique(y, return_counts=True)
        # calculate class prior probability
        for c, num in zip(unique_class, class_count):
            self._dic_class_prior[c] = num / sample_num

    def _cal_x_prob(self, x, y):
        """
        calculate Conditional Probability: P( xj | y=ck )
        like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
        for example the below ,as to class 1 feature 0 has 3 values "1, 2 , 3"
        the corresponding probability 0.22, 0.33, 0.44
        p( x1 = 1 | y = 1 ) = 0.22
        p( x1 = 2 | y = 1 ) = 0.33
        p( x1 = 3 | y = 1 ) = 0.44
        { 1: {0: {1: 0.22, 2: 0.33, 3: 0.44}, 1: {4: 0.11, 5: 0.44, 6: 0.44}},
         -1: {0: {1: 0.50, 2: 0.33, 3: 0.16}, 1: {4: 0.50, 5: 0.33, 6: 0.16}}}
        """
        unique_class = np.unique(y)
        for c in unique_class:
            self._cd_prob[c] = {}
            c_idx = np.where(y == c)[0]
            for i, col_feature in enumerate(x.T):
                dic_f_prob = {}
                self._cd_prob[c][i] = dic_f_prob
                for idx in c_idx:
                    if col_feature[idx] in dic_f_prob:
                        dic_f_prob[col_feature[idx]] += 1
                    else:
                        dic_f_prob[col_feature[idx]] = 1
                for k in dic_f_prob:
                    dic_f_prob[k] = dic_f_prob[k] * 1.0 / len(c_idx)

    def _pred_once(self, x):
        dic_ret = {}
        for y in self._dic_class_prior:
            y_prob = self._dic_class_prior[y]
            for i, v in enumerate(x):
                try:
                    y_prob = y_prob * self._cd_prob[y][i][v]
                except KeyError:
                    y_prob = y_prob * np.finfo(float).eps
            dic_ret[y] = y_prob
        return dic_ret

    def predict(self, x):
        if x.ndim == 1:
            return self._pred_once(x)
        else:
            labels = []
            for i in range(x.shape[0]):
                labels.append(self._pred_once(x[i]))
        return labels

    def get_class_prior(self):
        return self._dic_class_prior

    def get_cd_prob(self):
        return self._cd_prob


if __name__ == "__main__":
    sample_ratio = 0.05
    percentage_positive_instances_train = sample_ratio
    percentage_negative_instances_train = sample_ratio

    percentage_positive_instances_test = sample_ratio
    percentage_negative_instances_test = sample_ratio

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    with open('vocab.txt', 'w', encoding='UTF-8') as f:
        for word in vocab:
            f.write("%s\n" % word)
    print("Vocabulary (training set):", len(vocab))

    vocab = list(vocab)
    pos_train_vec = [words2vec(vocab, pos_train[index]) for index in range(0, len(pos_train))]
    pos_train_label = [1] * len(pos_train)
    neg_train_vec = [words2vec(vocab, neg_train[index]) for index in range(0, len(neg_train))]
    neg_train_label = [0] * len(neg_train)
    train_vec = pos_train_vec + neg_train_vec
    train_label = pos_train_label + neg_train_label
    x = np.array(train_vec)
    y = np.array(train_label)

    mnb = MultinomialNB()
    mnb.fit(x, y)

    item = np.array(words2vec(vocab, neg_test[10]))
    print(mnb.predict(item))
