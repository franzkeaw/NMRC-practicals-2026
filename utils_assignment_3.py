import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import accuracy_score


def hardlim(x):
    return np.asarray(x > 0).astype(int)


class Neuron:
    def __init__(self, inputs, targets, bias=None, weights=None,
                 learning_rate=1):
        self.inputs = inputs
        self.targets = targets
        self.n_inputs = inputs.shape[1]
        self.n_samples = inputs.shape[0]
        self.activation_function = hardlim
        self.learning_rate = learning_rate

        self.weights = weights if weights is not None else np.random.normal(0,
                                                                            0.01,
                                                                            self.n_inputs)
        self.bias = bias if bias is not None else np.random.normal(0, 0.01)

        self.accuracy = []

    def output(self, input):
        x = np.sum(input * self.weights) + self.bias
        return self.activation_function(x)

    def classify(self, inputs):
        return np.array([self.output(inp) for inp in inputs])

    def evaluate_accuracy(self, inputs, targets):
        return accuracy_score(targets, self.classify(inputs)) * 100

    def train_step(self, i):
        pattern, target = self.inputs[i], self.targets[i]
        err = self.output(pattern) - target
        self.weights -= self.learning_rate * err * pattern
        self.bias -= self.learning_rate * err
        self.accuracy.append(self.evaluate_accuracy(self.inputs, self.targets))


def plot_data_boundary_accuracy(neuron, X, y, j=None):
    weights, bias = neuron.weights, neuron.bias
    plot_line = lambda x: (-bias - weights[0] * x) / (weights[1] + 1e-5)

    f, ax = plt.subplots(1, 2, figsize=[15, 5])
    ax[0].scatter(X[y == 0, 0], X[y == 0, 1], color='g', s=40)
    ax[0].scatter(X[y == 1, 0], X[y == 1, 1], color='r', s=40)

    if j is not None:
        ax[0].scatter(X[j, 0], X[j, 1], facecolor=sns.xkcd_rgb['bright yellow'],
                      edgecolor='k', s=100, lw=2)

    x1, x2 = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y1, y2 = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx = np.arange(x1, x2, step=0.001)

    ax[0].plot(xx, plot_line(xx), c='k')
    ax[0].set_xlim([x1, x2])
    ax[0].set_ylim([y1, y2])
    ax[0].set_title('Decision boundary')

    epochs = [k + 1 for k in range(len(neuron.accuracy))]
    ax[1].scatter(epochs, neuron.accuracy, c='k')
    ax[1].plot(epochs, neuron.accuracy, c='k')
    ax[1].set_ylim([50, 100])
    ax[1].set_title('Training accuracy')
    ax[1].set_xticks(epochs)
    ax[1].set_yticks(range(int(10 * (np.min(neuron.accuracy) // 10)), 110, 10))

    plt.show()


class mlp_simulation:
    def __init__(self, X, y, learning_rate=1):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.out = widgets.Output()
        display(self.out)  # display output widget once

    def initialize_button(self, b):
        self.neuron = Neuron(self.X, self.y, learning_rate=self.learning_rate)
        self.neuron.accuracy.append(
            self.neuron.evaluate_accuracy(self.X, self.y))
        with self.out:
            clear_output(wait=True)
            plot_data_boundary_accuracy(self.neuron, self.X, self.y)

    def train_button(self, b):
        with self.out:
            for j in range(self.neuron.n_samples):
                self.neuron.train_step(j)
                clear_output(wait=True)
                plot_data_boundary_accuracy(self.neuron, self.X, self.y, j=j)

    def start(self):
        button_init = widgets.Button(description="Initialize")
        button_train = widgets.Button(description="Train")
        display(button_init, button_train)
        button_init.on_click(self.initialize_button)
        button_train.on_click(self.train_button)