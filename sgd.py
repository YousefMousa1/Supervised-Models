#################################
# Your name: YousefMousa
#################################
import matplotlib
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    n, d = data.shape
    w = np.zeros(d)

    for t in range(1, T + 1):
        eta_t = eta_0 / t
        i = np.random.randint(n)
        x_i = data[i]
        y_i = labels[i]

        if y_i * np.dot(w, x_i) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w
    return w

def compute_accuracy(w, b, data, labels):
    preds = np.sign(data @ w + b)
    return np.mean(preds == labels)

def train_and_validate(data, labels, val_data, val_labels, C, eta_0_values, T):
    val_accuracies = []
    for eta_0 in eta_0_values:
        accs = []
        for _ in range(10):
            w = SGD_hinge(data, labels, C, eta_0, T)
            acc = compute_accuracy(w, 0, val_data, val_labels)
            accs.append(acc)
        val_accuracies.append(np.mean(accs))
    return val_accuracies

def cross_validate_C(data, labels, val_data, val_labels, eta_0, C_values, T):
    val_accuracies = []
    for C in C_values:
        accs = []
        for _ in range(10):
            w = SGD_hinge(data, labels, C, eta_0, T)
            acc = compute_accuracy(w, 0, val_data, val_labels)
            accs.append(acc)
        val_accuracies.append(np.mean(accs))
    return val_accuracies

def visualize_weights(w):
    plt.imshow(w.reshape((28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.title("Visualization of weights w")
    plt.show()

train_X, train_y, val_X, val_y, test_X, test_y = helper()
#SECTION A WE GET THAT C = 1 IS BEST WITH AVG ACC = 0.9806
eta_0_values = [10**i for i in range(-5, 3)]
C_best = 1.0        # example value
T = 1000           # as required
avg_accuracies = []



for eta_0 in eta_0_values:
    accs = []
    for _ in range(10):
        w = SGD_hinge(train_X, train_y, C_best, eta_0, T)
        acc = compute_accuracy(w, 0, val_X, val_y)
        accs.append(acc)
    avg_accuracies.append(np.mean(accs))
    print(f"eta_0={eta_0}, avg accuracy={np.mean(accs):.4f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.semilogx(eta_0_values, avg_accuracies, marker='o')
plt.xlabel("η₀ (log scale)")
plt.ylabel("Average Validation Accuracy")
plt.title("Validation Accuracy vs η₀ (T=1000, C=1)")
plt.grid(True)
plt.tight_layout()
plt.show()


#Best parameters are C=1, eta_0 = 1 and T=1000, if you run the code you'll see accuracy of 0.9807.
#This is the best amongst all
#################################
