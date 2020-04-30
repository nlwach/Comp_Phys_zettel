import numpy as np
import matplotlib.pyplot as plt

def integrand(n, a):
    return x**n/(x+a)


def plot_function(x, a, n):
    for i in n:
        plt.plot(x, integrand(i, 5))
    plt.show()

def y_n(n, a, y_1):
    return 1/n - a* y_1

def y_n_rev(n, a, y_1):
    return -1/a * (y_1 - 1/(n))

def Iteration(n_0, n_1, a, y_0):
    res = []
    if n_0 == n_1:
        print("n_0 and n_1 are the same integers!")
        return
    if n_0 < n_1:
        for n in range(n_0 + 1, n_1 + 1):
            value = y_n(n, a, y_0)
            y_0 = value
    else:
        for n in range(n_0, n_1, -1):
            y_0 = y_n_rev(n, a, y_0)
    print(y_0)

plot_function(np.linspace(0, 1, 1000), 4, [1,5,10,20,30,50])

a = 5
y_0 = np.log((1+a)/a)

Iteration(50,30,5,1)


