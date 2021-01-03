import tensorflow as tf
import numpy as np
from typing import NamedTuple
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# TASK CONFIG
ACTIVATION_DENSE = 'softmax'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']
N_CLASSES = 10
SHAPE = (28, 28, 1)

ACTIVATION_CONV2D = 'relu'
EPOCHS = 5
OPTIMIZER = 'adam'


class Data(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class Architecture(NamedTuple):
    n_neurons: int
    dropout_rate: float


LABELS_FASHION_MNIST = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


def fix_data(data) -> Data:
    (x_train, y_train), (x_test, y_test) = data
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return Data(x_train, y_train, x_test, y_test)


def architecture_gen():
    for n_neurons in range(30, 35):
        for dropout_rate in np.linspace(0.3, 0.7, 5):
            yield Architecture(n_neurons, dropout_rate)


def display_matrix(y_true, y_pred):
    display = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=LABELS_FASHION_MNIST.values())
    fig = plt.figure(figsize=(14, 9))
    display.plot(ax=fig.add_subplot(111), values_format="d")
    plt.show()


def build_model(architecture: Architecture) -> tf.keras.Sequential:
    n_neurons, dropout_rate = architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=SHAPE))
    model.add(tf.keras.layers.Conv2D(n_neurons, kernel_size=2, activation=ACTIVATION_CONV2D))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(n_neurons, kernel_size=3, activation=ACTIVATION_CONV2D))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(N_CLASSES, activation=ACTIVATION_DENSE))
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)
    return model


def fit(model: tf.keras.Sequential, data: Data) -> tf.keras.Sequential:
    x_train, y_train, x_test, y_test = data
    model.fit(x_train, y_train, epochs=1, verbose=1, validation_data=(x_test, y_test))
    return model


def show_architecture_on_fashion_mnist(architecture: Architecture):
    data = fix_data(tf.keras.datasets.fashion_mnist.load_data())

    print("Running %s on Fashion-MNIST..." % str(architecture))
    model = fit(build_model(architecture), data)
    y_pred = np.argmax(model.predict(data.x_test), axis=-1)
    display_matrix(data.y_test, y_pred)
    print("Accuracy: %.6f" % accuracy_score(data.y_test, y_pred))


def run_architecture(architecture, data):
    print("Running %s on MNIST..." % str(architecture))
    loss, accuracy = fit(build_model(architecture), data).evaluate(data.x_test, data.y_test, verbose=0)
    print("Accuracy: %.6f. Loss: %.2f" % (accuracy, loss), end="\n\n")
    return accuracy


def find_best_architecture_on_mnist() -> Architecture:
    data = fix_data(tf.keras.datasets.mnist.load_data(path="mnist.npz"))
    return max(architecture_gen(), key=lambda arch: run_architecture(arch, data))


if __name__ == '__main__':
    show_architecture_on_fashion_mnist(find_best_architecture_on_mnist())
