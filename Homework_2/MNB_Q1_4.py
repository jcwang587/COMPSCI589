from collections import Counter, defaultdict
import numpy as np
import math
from utils import *


class MultinomialNaiveBayes:
    def __init__(self, classes):
        self.classes = classes

    def group_by_class(self, X, y):
        data = dict()
        for c in self.classes:
            data[c] = X[np.where(y == c)]
        return data

    def fit(self, X, y):
        self.n_class_items = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()
        n = len(X)
        grouped_data = self.group_by_class(X, y)
        for c, data in grouped_data.items():
            self.n_class_items[c] = len(data)
            self.log_class_priors[c] = math.log(self.n_class_items[c] / n)
            self.word_counts[c] = defaultdict(lambda: 0)
            for text in data:
                counts = Counter(text)
                for word, count in counts.items():
                    if word not in self.vocab:
                        self.vocab.add(word)
                    self.word_counts[c][word] += count
        return self

    def laplace_smoothing(self, word, text_class):
        num = self.word_counts[text_class][word] + 1
        denom = self.n_class_items[text_class] + len(self.vocab)
        return math.log(num / denom)

    def predict(self, X):
        result = []
        for text in X:
            class_scores = {c: self.log_class_priors[c] for c in self.classes}
            print(X)
            words = set(text)
            for word in words:
                if word not in self.vocab: continue
                for c in self.classes:
                    log_w_given_c = self.laplace_smoothing(word, c)
                    class_scores[c] += log_w_given_c
            result.append(max(class_scores, key=class_scores.get))
        return result


if __name__ == "__main__":
    sample_ratio = 0.004
    percent_positive_instance_train = sample_ratio
    percent_negative_instance_train = sample_ratio
    percent_positive_instance_test = sample_ratio
    percent_negative_instance_test = sample_ratio

    (pos_train, neg_train, vocab) = load_training_set(percent_positive_instance_train, percent_negative_instance_train)
    (pos_test, neg_test) = load_test_set(percent_positive_instance_test, percent_negative_instance_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    with open('vocab.txt', 'w', encoding='UTF-8') as f:
        for word in vocab:
            f.write("%s\n" % word)
    print("Vocabulary (training set):", len(vocab))

    vocab = list(vocab)
    pos_train_label = [1] * len(pos_train)
    neg_train_label = [0] * len(neg_train)
    train_vec = pos_train + neg_train
    train_label = pos_train_label + neg_train_label
    x = np.array(train_vec)
    y = np.array(train_label)

    MNB = MultinomialNaiveBayes(classes=np.unique(y)).fit(x, y)
    item = np.array(neg_test[10])
    print(MNB.predict(item))
