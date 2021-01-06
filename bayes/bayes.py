import os
import numpy as np
from typing import Tuple, List
from collections import deque
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

Subject = List[int]
Body = List[int]
Is_Legit = int
Message = Tuple[Subject, Body, Is_Legit]

# words, size
Words = List[int]
Size = int
NGram = Tuple[Words, Size]

NGRAM_SIZE = 3
N_PARTS = 2
N_CLASSES = 2
PREFIX_SUBJECT_STR_LENGTH = 9


# NGram

def hash_ngram(ngram: NGram) -> str:
    return '_'.join(map(str, ngram[0]))


def get_message_ngram(subject_words: List[int], body_words: List[int], n=NGRAM_SIZE) -> [NGram]:
    last_words = deque()
    for words in [subject_words, body_words]:
        for i in range(len(words)):
            last_words.append(words[i])
            if i < n - 1:
                continue
            n_gram = deque_to_list(last_words)
            last_words.popleft()
            yield n_gram, n


def deque_to_list(deq) -> List[int]:
    res = []
    for i in range(len(deq)):
        res.append(deq[i])
    return res


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
        # with open('likelihood.npy', 'rb') as f:
        #     self.likelihood = np.load(f)

        self.likelihood = self._calc_likelihood(X, y)
        with open('likelihood.npy', 'wb') as f:
            np.save(f, self.likelihood)

    def predict_sample_with_proba(self, x) -> Tuple[int, float]:
        proba = list(map(lambda class_idx: self._bayes_formula(x, class_idx), range(self.n_classes)))
        return np.argmax(proba), proba[0] / sum(proba)

    def predict_with_proba(self, X) -> Tuple[List[int], List[float]]:
        predicts_with_proba = list(map(self.predict_sample_with_proba, X))
        predicts = []
        probas = []
        for predict, proba in predicts_with_proba:
            predicts.append(predict)
            probas.append(proba)
        return predicts, probas

    def _bayes_formula(self, x, class_idx):
        acc = 0
        for feature_idx in range(len(x)):
            prob = self.likelihood[class_idx][feature_idx]
            if x[feature_idx] == 0:
                prob = 1 - prob
            acc += np.log(prob)
        return np.log(self.penalties[class_idx] * self.prior_probability[class_idx]) + acc

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


def show_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-curve')
    plt.legend(loc="lower right")
    plt.show()


def calc_false_negative(y_true, y_pred):
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return FN


if __name__ == '__main__':
    parts = read_messages()

    lambda_legit = 10000
    lambda_spam = 1
    alpha = 0.1

    clf = NaiveBayesClassifier(N_CLASSES, [lambda_spam, lambda_legit], alpha)

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
        y_pred, y_score = clf.predict_with_proba(X_test)

        print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
        show_roc_curve(y_test, y_score)
        print("False negative: %d" % calc_false_negative(y_test, y_pred))
