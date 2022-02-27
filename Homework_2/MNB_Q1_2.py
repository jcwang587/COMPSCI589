import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils import *
import pprint
from collections import Counter, defaultdict


def words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for words in input_set:
        if words in vocab_list:
            return_vec[vocab_list.index(words)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % words)
    return return_vec


class NBayes:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_  # 贝叶斯估计方法参数lambda
        self.p_prior = {}  # 模型的先验概率, 注意这里的先验概率不是指预先人为设定的先验概率，而是需要估计的P(y=Ck)
        self.p_condition = {}  # 模型的条件概率

    def fit(self, X_data, y_data):
        N = y_data.shape[0]
        # 后验期望估计P(y=Ck)的后验概率，设定先验概率为均匀分布
        c_y = Counter(y_data)
        K = len(c_y)
        for key, val in c_y.items():
            self.p_prior[key] = (val + self.lambda_) / (N + K * self.lambda_)
        # 后验期望估计P(Xd=a|y=Ck)的后验概率，同样先验概率为均匀分布
        for d in range(X_data.shape[1]):  # 对各个维度分别进行处理
            Xd_y = defaultdict(int)
            vector = X_data[:, d]
            Sd = len(np.unique(vector))
            for xd, y in zip(vector, y_data):  # 这里Xd仅考虑出现在数据集D中的情况，故即使用极大似然估计叶没有概率为0的情况
                Xd_y[(xd, y)] += 1
            for key, val in Xd_y.items():
                self.p_condition[(d, key[0], key[1])] = (val + self.lambda_) / (c_y[key[1]] + Sd * self.lambda_)
        return

    def predict(self, X):
        p_post = defaultdict()
        for y, py in self.p_prior.items():
            p_joint = py  # 联合概率分布
            for d, Xd in enumerate(X):
                p_joint *= self.p_condition[(d, Xd, y)]  # 条件独立性假设
            p_post[y] = p_joint  # 分母P(X)相同，故直接存储联合概率分布即可
        return max(p_post, key=p_post.get)


if __name__ == "__main__":
    percentage_positive_instances_train = 0.0004
    percentage_negative_instances_train = 0.0004

    percentage_positive_instances_test = 0.0004
    percentage_negative_instances_test = 0.0004

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

    clf = NBayes(lambda_=1)
    clf.fit(x, y)
    print(clf.p_prior)

    t = np.array(words2vec(vocab, pos_test[0]))
    print(clf.predict(t))
