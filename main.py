# A 2-1-MLP (Multi-Layer-Perzeptron)
# Hidden Layer: hyperbolic tangent
# Training via delta-rule
# Error function: The sum of the squared differences
# learning rate 0.1 >= eta >= 0.01. Later 0.01 >= eta >= 0.001 (lecture 03, s. 78)
import os

import numpy as np
import matplotlib.pyplot as plt


def init_weights():
    # see lecture 03 slide 84
    # -0.1 <= w_ij <= +0.1 OR -0.5 <= w_ij <= +0.5

    # np.random.rand returns random samples from a uniform distribution over [0,1)
    return np.random.rand(3, 1) * 0.2 - 0.1


def eval(x1, x2, fix_W_ij):
    global net_j
    global o_j

    net_j = np.dot(np.array([x1, x2, 1]), fix_W_ij)
    o_j = np.tanh(net_j)
    return o_j


def train(x_train, y_train):
    global W_ij

    assert x_train.shape[0] == y_train.shape[0]
    for i in range(x_train.shape[0]):
        # This updates net_j and o_j
        eval(x_train[i][0], x_train[i][1], W_ij)

        # delta rules lecture 03 slide 123
        # for output neuron m

        # The derivation of hyperbolic tangent
        # tanh'(z) = 1.0 - tanh^2(z)
        # lecture 03 slide 53

        delta_j = (o_j - y_train[i]) * (1.0 - o_j ** 2)
        # Delta_W_ij lecture 03 slide 123
        # Delta_W_ij = eta * delta_j * out_i
        Delta_W_ij = eta * delta_j * np.array([x_train[i][0], x_train[i][1], 1]).reshape((3, 1))

        # Adjust weights
        W_ij += Delta_W_ij


def plot_result(X, Y):
    # Plot result
    # 1. Plot data
    plt.scatter(X[:, 0], X[:, 1], c=Y)

    # 2. Plot seperating line (see lecture 09 slide 39)
    # w1z1+w2z2+b=0
    # z2= -(w1/w2)*z1 -b/w2
    # TODO handle case w2=0
    (w1, w2, b) = (W_ij.item(0), W_ij.item(1), W_ij.item(2))

    # TODO Plot not only the function in [0,10], but infinitely long
    z1 = np.linspace(-10, 10, 100)
    z2 = -(w1 / w2) * z1 - b / w2
    plt.plot(z1, z2, '-r', label=f"z2 = -(w1/w2)*z1 -b/w2, w1={round(w1, 2)}, w2={round(w2, 2)}, b={round(b, 2)}")
    plt.title(f"Graph of z2 = -(w1/w2)*z1 -b/w2, w1={round(w1, 2)}, w2={round(w2, 2)}, b={round(b, 2)}")
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()

    plt.savefig(f"out/plot_result.png")
    plt.show()


# lecture 02 slide 67
# sum of squared errors
def error(pX, pY, fix_W_ij):
    return 1 / 2 * (eval(pX[0], pX[1], fix_W_ij) - pY) ** 2


def global_error(X, Y, fix_W_ij):
    assert X.shape[0] == Y.shape[0]
    err = 0
    for i in range(Y.shape[0]):
        err += error(X[i], Y[i], fix_W_ij)
    return err


def call_global_error(X, Y, w0: float, w1: float, w2: float):
    return global_error(X, Y, np.array([w0, w1, w2]))


def plot_error_surface_0(x_train, y_train, w0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    w1 = w2 = np.arange(-10.0, 10.0, 0.05)
    W1, W2 = np.meshgrid(w1, w2)
    zs = np.array(call_global_error(x_train, y_train, w0, np.ravel(W1), np.ravel(W2)))
    Z = zs.reshape(W1.shape)

    ax.plot_surface(W1, W2, Z)

    plt.title(f"sum of squared errors/2, w0={w0}")
    ax.set_xlabel('W1 Label')
    ax.set_ylabel('W2 Label')
    ax.set_zlabel('Global error')

    plt.savefig(f"out/plot_error_surface_fix_w0.png")
    plt.show()


def plot_error_surface_1(x_train, y_train, w1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    w0 = w2 = np.arange(-10.0, 10.0, 0.05)
    W0, W2 = np.meshgrid(w0, w2)
    zs = np.array(call_global_error(x_train, y_train, np.ravel(W0), w1, np.ravel(W2)))
    Z = zs.reshape(W0.shape)

    ax.plot_surface(W0, W2, Z)

    plt.title(f"sum of squared errors/2, w1={w1}")
    ax.set_xlabel('W0 Label')
    ax.set_ylabel('W2 Label')
    ax.set_zlabel('Global error')

    plt.savefig(f"out/plot_error_surface_fix_w1.png")
    plt.show()


def plot_error_surface_2(x_train, y_train, w2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    w0 = w1 = np.arange(-10.0, 10.0, 0.05)
    W0, W1 = np.meshgrid(w0, w1)
    zs = np.array(call_global_error(x_train, y_train, np.ravel(W0), np.ravel(W1), w2))
    Z = zs.reshape(W0.shape)

    ax.plot_surface(W0, W1, Z)

    plt.title(f"sum of squared errors/2, w2={w2}")
    ax.set_xlabel('W0 Label')
    ax.set_ylabel('W1 Label')
    ax.set_zlabel('Global error')

    plt.savefig(f"out/plot_error_surface_fix_w2.png")
    plt.show()

# This is for testing purposes only
def classify(pX):
    # define line ax + b. b=0
    a, b = 0.5, -1
    if a*pX[0]+b <= pX[1]:
        return -1
    else:
        return 1

# This is for testing purposes only
def generate_random_input(n):
    x_train = np.random.randint(20, size=(n, 2))-10
    y_train = np.zeros(n)
    for i in range(n):
        y_train[i] = classify(np.ravel(x_train[i]))
    return x_train, y_train


if __name__ == '__main__':
    global eta
    # Choose variables
    eta = 0.1

    # Load x_train
    #data = np.loadtxt('PA-F-train.txt', comments="#")
    #x_train = data[:, :-1]
    #y_train = data[:, -1]

    # Randomly generate data for testing purposes
    x_train, y_train = generate_random_input(20)

    # initialize weights
    W_ij = init_weights()

    # train model
    train(x_train, y_train)

    # Plot output
    # Create output dir for the output neurons and plots
    path = 'out'
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)
    plot_result(x_train, y_train)
    plot_error_surface_0(x_train, y_train, W_ij[0])
    plot_error_surface_1(x_train, y_train, W_ij[1])
    plot_error_surface_2(x_train, y_train, W_ij[2])
