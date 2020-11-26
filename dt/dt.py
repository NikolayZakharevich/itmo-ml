from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


def read_csv(name: str):
    data = pd.read_csv(name)
    return data[data.columns[:-1]].loc[:].values, data[data.columns[-1]].loc[:].values


def read_dataset(num: int, suffix: str):
    return read_csv("DT_csv/%s_%s.csv" % (str(num).zfill(2), suffix))


def hyper_params_with_max_height_gen():
    for criterion in ['gini', 'entropy']:
        for splitter in ['best', 'random']:
            for max_depth in [x for x in range(1, 30)]:
                yield (criterion, splitter, max_depth)


def hyper_params_with_max_features_gen(n_features: int):
    for criterion in ['gini', 'entropy']:
        for splitter in ['best', 'random']:
            for max_features in [x for x in range(1, min(n_features, 25))]:
                yield (criterion, splitter, max_features)


def train(X, Y, hyper_params):
    criterion, splitter, max_depth = hyper_params
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    return clf.fit(X, Y)


def find_best_hyper_params(X_train, Y_train, X_test, Y_test):
    best_score = None
    best_hyper_params = None

    for hyper_params in hyper_params_with_max_height_gen():
        clf = train(X_train, Y_train, hyper_params)
        score = clf.score(X_test, Y_test)
        if best_score is None or score > best_score:
            best_score = score
            best_hyper_params = hyper_params

    return (best_score, best_hyper_params)


def get_accuracy_all(dataset_num, best_hyper_params, height_all):
    best_criterion, best_splitter, _ = best_hyper_params
    X_train, Y_train = read_dataset(dataset_num, 'train')
    X_test, Y_test = read_dataset(dataset_num, 'test')
    result = []
    for height in height_all:
        result.append(train(X_train, Y_train, (best_criterion, best_splitter, height)).score(X_test, Y_test))
    return result


def show_accuracy_on_height(dataset1_num, best_hyper_params1, dataset2_num, best_hyper_params2):
    height_all = [x for x in range(1, 30)]
    accuracy1_all = get_accuracy_all(dataset1_num, best_hyper_params1, height_all)
    accuracy2_all = get_accuracy_all(dataset2_num, best_hyper_params2, height_all)

    plt.xlabel('Max height', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy dependence on height')
    plt.plot(height_all, accuracy1_all, label='Dataset #%d' % dataset1_num)
    plt.plot(height_all, accuracy2_all, label='Dataset #%d' % dataset2_num)
    plt.legend()
    plt.show()


def dt_forest(X_train, Y_train, X_test, Y_test):
    predicts = []
    for hyper_params in hyper_params_with_max_features_gen(len(X_train[0])):
        predicts.append(train(X_train, Y_train, hyper_params).predict(X_test))

    Y_predicted = []
    for i in range(len(X_test)):
        Y_predicted.append(Counter(y_pred[i] for y_pred in predicts).most_common(1)[0][0])

    return accuracy_score(Y_test, Y_predicted, normalize=True)


def run_part1():
    min_height, min_height_dataset_args = None, (None, None)
    max_height, max_height_dataset_args = None, (None, None)
    for num in range(20, 22):
        X_train, Y_train = read_dataset(num, 'train')
        X_test, Y_test = read_dataset(num, 'test')

        accuracy, best_hyper_params = find_best_hyper_params(X_train, Y_train, X_test, Y_test)
        _, _, height = best_hyper_params

        if min_height is None or height < min_height:
            min_height = height
            min_height_dataset_args = (num, best_hyper_params)

        if max_height is None or height > max_height:
            max_height = height
            max_height_dataset_args = (num, best_hyper_params)
        print("Dataset #%d: " % num, accuracy, best_hyper_params)

    show_accuracy_on_height(*min_height_dataset_args, *max_height_dataset_args)


def run_part2():
    for num in range(1, 22):
        X_train, Y_train = read_dataset(num, 'train')
        X_test, Y_test = read_dataset(num, 'test')
        print(
            "Dataset #%d: " % num,
            dt_forest(X_train, Y_train, X_train, Y_train),
            dt_forest(X_train, Y_train, X_test, Y_test)
        )


if __name__ == '__main__':
    # run_part1()
    run_part2()

    ''' best hyper params for each dataset:
Dataset #1:  0.9997429966589566 ('entropy', 'best', 3)
Dataset #2:  0.7040712468193384 ('entropy', 'best', 9)
Dataset #3:  1.0 ('gini', 'best', 1)
Dataset #4:  0.992 ('entropy', 'best', 5)
Dataset #5:  0.9956709956709957 ('gini', 'best', 1)
Dataset #6:  0.9988962472406181 ('entropy', 'best', 3)
Dataset #7:  0.9967441860465116 ('entropy', 'best', 3)
Dataset #8:  0.997920997920998 ('gini', 'best', 2)
Dataset #9:  0.8392156862745098 ('entropy', 'best', 5)
Dataset #10:  0.9979879275653923 ('entropy', 'best', 4)
Dataset #11:  0.999195171026157 ('gini', 'best', 1)
Dataset #12:  0.8776312096547852 ('entropy', 'best', 10)
Dataset #13:  0.6486238532110091 ('entropy', 'best', 7)
Dataset #14:  0.990351215746816 ('entropy', 'best', 5)
Dataset #15:  1.0 ('gini', 'best', 1)
Dataset #16:  1.0 ('gini', 'best', 1)
Dataset #17:  0.8453378001116695 ('entropy', 'best', 7)
Dataset #18:  0.9426656738644825 ('entropy', 'best', 5)
Dataset #19:  0.8342085521380345 ('entropy', 'best', 7)
Dataset #20:  0.9706814580031695 ('entropy', 'best', 7)
Dataset #21:  0.8097653772986684 ('entropy', 'best', 22)

min height: 1 (dataset 03)
min height: 17 (dataset 21)

dt forest results: 
Dataset #1:  0.941917244924184 0.9920328964276536
Dataset #2:  0.9791348600508906 0.6463104325699746
Dataset #3:  0.9696239151398264 1.0
Dataset #4:  0.9826086956521739 0.9954782608695653
Dataset #5:  0.9761904761904762 0.9956709956709957
Dataset #6:  0.989514348785872 0.9950331125827815
Dataset #7:  0.9832558139534884 0.9958139534883721
Dataset #8:  1.0 0.997920997920998
Dataset #9:  1.0 0.8137254901960784
Dataset #10:  0.9452716297786721 0.9979879275653923
Dataset #11:  0.9605633802816902 0.999195171026157
Dataset #12:  0.7227055851810272 0.7204602862756104
Dataset #13:  0.9944954128440368 0.6688073394495413
Dataset #14:  0.987263604785797 0.991509069857198
Dataset #15:  0.9591591591591592 0.9993993993993994
Dataset #16:  0.9610591900311527 1.0
Dataset #17:  0.9988833054159687 0.8464544946957008
Dataset #18:  0.9932985852568875 0.9314966492926284
Dataset #19:  0.90847711927982 0.7419354838709677
Dataset #20:  0.7729793977812995 0.8094294770206022
Dataset #21:  0.5199746353836399 0.5168040583386176
'''
