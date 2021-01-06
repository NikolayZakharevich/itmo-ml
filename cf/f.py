import numpy as np
from typing import List

Q = 2
feature_idx_map = {}


def bernulli_with_laplace_smoothing(x_count, y_count, alpha=1) -> float:
    return (x_count + alpha) / (y_count + alpha * Q)


def vectorize(messages: List[List[str]]):
    word_dict = {}
    for i, message in enumerate(messages):
        for word in message:
            if word not in word_dict:
                feature_idx_map[word] = len(word_dict)
                word_dict[word] = set()
            word_dict[word].add(i)

    X = np.zeros((len(messages), len(word_dict)), dtype=int)
    for word_idx, word_messages_indexes in enumerate(word_dict.values()):
        for word_message_idx in word_messages_indexes:
            X[word_message_idx][word_idx] = 1
    return X


class NaiveBayesClassifier():
    n_classes = None
    penalties = None
    alpha = None

    prior_probability = None
    likelihood = None

    def __init__(self, n_classes, penalties, alpha=1):
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
        return np.vectorize(self.predict_sample)(X)

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
    K = int(input())
    penalties = list(map(int, input().split()))
    alpha = int(input())
    N = int(input())
    messages = []

    y = []
    for _ in range(N):
        input_str = input().split()
        y.append(int(input_str[0]) - 1)
        messages.append(input_str[2:])

    X = vectorize(messages)
    n_features = len(X[0])

    clf = NaiveBayesClassifier(K, penalties, alpha)
    clf.fit(X, y)

    M = int(input())
    for _ in range(M):
        x_test = np.zeros(n_features)
        for word in input().split()[1:]:
            feature_idx = feature_idx_map.get(word, -1)
            if feature_idx != -1:
                x_test[feature_idx] = 1
        print(*clf.predict_sample(x_test))
