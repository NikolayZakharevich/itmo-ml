import os
import numpy as np
from typing import Tuple, List
from sklearn.metrics import accuracy_score

from ngram import get_message_ngram, hash_ngram

# body
Subject = List[int]
Body = List[int]
Is_Legit = int
Message = Tuple[Subject, Body, Is_Legit]

N_PARTS = 2
N_CLASSES = 2
PREFIX_SUBJECT_STR_LENGTH = 9


# Read data

def read_message(filename: str) -> Message:
    with open(filename, 'r') as file:
        subject_words = file.readline()[PREFIX_SUBJECT_STR_LENGTH:-1].split()
        file.readline()
        body_words = file.readline()[:-1].split()
    return list(map(int, subject_words)), list(map(int, body_words)), int("legit" in filename)


def read_messages(messages_dirname='messages/') -> List[Tuple[np.ndarray, np.ndarray]]:
    word_dict = {}

    X_all = []
    y_all = []
    samples_cnt_per_number = []
    for number in range(N_PARTS):
        y = []
        directory = messages_dirname + 'part' + str(number + 1)

        samples_cnt = 0
        for i, filename in enumerate(os.listdir(directory)):
            subject_words, body_words, is_legit = read_message(directory + '/' + filename)
            y.append(is_legit)
            for ngram in get_message_ngram(subject_words, body_words):
                ngram_hash = hash_ngram(ngram)
                if ngram_hash not in word_dict:
                    word_dict[ngram_hash] = [set() for _ in range(N_PARTS)]
                word_dict[ngram_hash][number].add(i)
            samples_cnt += 1

        y_all.append(np.array(y))
        samples_cnt_per_number.append(samples_cnt)

    for number, samples_cnt in enumerate(samples_cnt_per_number):
        X = np.zeros((samples_cnt, len(word_dict)), dtype=int)
        for word_idx, word_messages_indexes in enumerate(word_dict.values()):
            for word_message_idx in word_messages_indexes[number]:
                X[word_message_idx][word_idx] = 1
        X_all.append(X)

    return list(zip(X_all, y_all))


# Bayes

def bernulli_with_laplace_smoothing(x_count, y_count, alpha=1.0):
    return (x_count + alpha) / (y_count + alpha * 2)


class NaiveBayesClassifier():
    n_classes = None
    penalties = None
    alpha = None

    prior_probability = None
    likelihood = None

    def __init__(self, n_classes, penalties, alpha: float = 1):
        self.n_classes = n_classes
        self.penalties = penalties
        self.alpha = alpha

    def fit(self, X, y):
        self.prior_probability = self._calc_prior_probability(y)
        self.likelihood = self._calc_likelihood(X, y)

    def predict_sample(self, x) -> List[float]:
        res = []
        sum = 0
        for class_idx in range(self.n_classes):
            temp = 1
            for feature_idx in range(len(x)):
                prob = self.likelihood[class_idx][feature_idx]
                if x[feature_idx] == 0:
                    prob = 1 - prob
                temp *= prob
            temp *= (self.penalties[class_idx] * self.prior_probability[class_idx])
            sum += temp
            res.append(temp)
        return np.array(res) / sum

    def predict(self, X) -> List[List[float]]:
        return np.vectorize(self.predict_sample)(X.tolist())

    def _calc_prior_probability(self, y):
        count_per_class = np.zeros(self.n_classes)
        for y_i in y:
            count_per_class[y_i] += 1
        return count_per_class / sum(count_per_class)

    def _calc_likelihood(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        samples_per_class = np.zeros(self.n_classes, dtype=int)
        for feature_idx in range(n_samples):
            samples_per_class[y[feature_idx]] += 1

        features_per_class = np.zeros((self.n_classes, n_features), dtype=int)
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                features_per_class[y[sample_idx]][feature_idx] += X[sample_idx][feature_idx] == 1

        res = np.zeros((self.n_classes, n_features))
        for class_idx in range(self.n_classes):
            for feature_idx in range(n_features):
                res[class_idx][feature_idx] = bernulli_with_laplace_smoothing(
                    features_per_class[class_idx][feature_idx],
                    samples_per_class[class_idx], self.alpha)
        return res


if __name__ == '__main__':
    parts = read_messages()

    lambda_legit = 1
    lambda_spam = 1
    alpha = 0.1

    clf = NaiveBayesClassifier(2, [lambda_spam, lambda_legit], alpha)

    # k-fold cross validation
    for test_idx in range(N_PARTS):
        X_test, y_test = parts[test_idx]

        X_train = []
        y_train = []
        for train_idx in range(N_PARTS):
            if test_idx == train_idx:
                continue
            X, y = parts[train_idx]
            X_train.extend(X)
            y_train.extend(y)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        clf.fit(X_train, y_train)
        print(accuracy_score(y_test, clf.predict(X_test)))
