import random

import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return x ** 4 * np.sin(x) + 1 / np.sqrt(x + 1)


def indefinite_integral(x):
    return np.sin(x) * (4 * x ** 3 - 24 * x) + np.cos(x) * (- x ** 4 + 12 * x ** 2 - 24) + 2 * np.sqrt(x + 1)


def definite_integral(a, b):
    return indefinite_integral(b) - indefinite_integral(a)


def function_xy(x, y):
    return y ** 4 * np.sin(x) + 1 / np.sqrt(x + 1)


def definite_integral_xy():
    return (273 - 243 * np.cos(3)) / 5                  # a = 0, b = 3, c = 0, d = 3


def rectangle_method(a, b, step):
    n = int((b - a) // step)
    remainder = (b - a) % step
    sum_ = 0
    for i in range(n):
        sum_ += function(a + i * step + step / 2)
    sum_ *= step
    sum_ += remainder * (function(b - remainder / 2))
    return sum_


def trapezoidal_method(a, b, step):
    n = int((b - a) // step)
    remainder = (b - a) % step
    sum_ = function(a) + function(b - remainder)
    for i in range(1, n):
        sum_ += 2 * function(a + i * step)
    sum_ *= step / 2
    sum_ += remainder / 2 * (function(b) + function(b - remainder))
    return sum_


def parabola_method(a, b, step):
    n = int((b - a) // step)
    remainder = (b - a) % step
    ex = n % 2
    remainder += ex * step
    remainder /= 2
    n -= ex
    n = int(n / 2)
    sum_ = function(a) + function(a + 2 * n * step)
    for i in range(1, n + 1):
        sum_ += 4 * function(a + (2 * i - 1) * step)
    for i in range(1, n):
        sum_ += 2 * function(a + 2 * i * step)
    sum_ *= step / 3
    sum_ += remainder / 3 * (function(b) + 4 * function(b - remainder) + function(b - 2 * remainder))
    return sum_


def cubic_parabola_method(a, b, step):
    n = int((b - a) // step)
    remainder = (b - a) % step
    ex = n % 3
    remainder += ex * step
    remainder /= 3
    n -= ex
    n = int(n / 3)
    sum_ = function(a) + function(a + 3 * n * step)
    for i in range(1, n + 1):
        sum_ += 3 * (function(a + (3 * i - 2) * step) + function(a + (3 * i - 1) * step))
    for i in range(1, n):
        sum_ += 2 * function(a + 3 * i * step)
    sum_ *= 3 * step / 8
    sum_ += 3 * remainder / 8 * (function(b) + 3 * (function(b - remainder) + function(b - 2 * remainder))
                                 + function(b - 3 * remainder))
    return sum_


def boole_method(a, b, step):
    n = int((b - a) // step)
    remainder = (b - a) % step
    ex = n % 4
    remainder += ex * step
    remainder /= 4
    n -= ex
    n = int(n / 4)
    sum_ = 7 * (function(a) + function(a + 4 * n * step))
    for i in range(1, n + 1):
        sum_ += 32 * (function(a + (4 * i - 3) * step) + function(a + (4 * i - 1) * step))
        sum_ += 12 * function(a + (4 * i - 2) * step)
    for i in range(1, n):
        sum_ += 14 * function(a + 4 * i * step)
    sum_ *= 2 * step / 45
    sum_ += 2 * remainder / 45 * (7 * (function(b) + function(b - 4 * remainder)) + 12 * function(b - 2 * remainder)
                                  + 32 * (function(b - remainder) + function(b - 3 * remainder)))
    return sum_


def gauss_method(a, b, n):
    if n == 2:
        return (b - a) / 2 * (function((a + b) / 2 - (b - a) / 2 * 0.5773)
                              + function((a + b) / 2 + (b - a) / 2 * 0.5773))
    if n == 3:
        return (b - a) / 2 * (0.5555 * function((a + b) / 2 - (b - a) / 2 * 0.7745)
                              + 0.8888 * function((a + b) / 2)
                              + 0.5555 * function((a + b) / 2 + (b - a) / 2 * 0.7745))
    if n == 4:
        return (b - a) / 2 * (0.3478 * function((a + b) / 2 - (b - a) / 2 * 0.8611)
                              + 0.6521 * function((a + b) / 2 - (b - a) / 2 * 0.3399)
                              + 0.6521 * function((a + b) / 2 + (b - a) / 2 * 0.3399)
                              + 0.3478 * function((a + b) / 2 + (b - a) / 2 * 0.8611))
    if n == 5:
        return (b - a) / 2 * (0.4786 * function((a + b) / 2 - (b - a) / 2 * 0.9061)
                              + 0.2369 * function((a + b) / 2 - (b - a) / 2 * 0.5384)
                              + 0.5688 * function((a + b) / 2)
                              + 0.2369 * function((a + b) / 2 + (b - a) / 2 * 0.5384)
                              + 0.4786 * function((a + b) / 2 + (b - a) / 2 * 0.9061))
    if n == 6:
        return (b - a) / 2 * (0.1713 * function((a + b) / 2 - (b - a) / 2 * 0.9324)
                              + 0.3607 * function((a + b) / 2 - (b - a) / 2 * 0.6612)
                              + 0.4679 * function((a + b) / 2 - (b - a) / 2 * 0.2386)
                              + 0.4679 * function((a + b) / 2 + (b - a) / 2 * 0.2386)
                              + 0.3607 * function((a + b) / 2 + (b - a) / 2 * 0.6612)
                              + 0.1713 * function((a + b) / 2 + (b - a) / 2 * 0.9324))


def monte_carlo_method(a, b, n):
    sum_ = 0
    for i in range(n):
        sum_ += function(random.uniform(a, b))
    return (b - a) * sum_ / n


def monte_carlo_method_xy(a, b, c, d, n):
    sum_ = 0
    for i in range(n):
        sum_ += function_xy(random.uniform(a, b), random.uniform(c, d))
    return (b - a) * (d - c) * sum_ / n


def draw_function():
    """
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)
    z = function(x_grid, y_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z',
           title='np.log(y) ** 2 / (np.tan(x) ** 2 + 2) - np.sin(np.log(x)) / (y ** 2 + 1)')
    ax.plot_surface(x_grid, y_grid, z, cmap='Wistia')

    plt.show()
    """
    values_x = []
    values_rectangle = []
    values_trapezoidal = []
    values_parabola = []
    values_cubic = []
    values_boole = []
    i = 1
    while i > 1e-3:
        values_x.append(i)
        values_rectangle.append(definite_integral(0, 3) - rectangle_method(0, 3, i))
        values_trapezoidal.append(definite_integral(0, 3) - trapezoidal_method(0, 3, i))
        values_parabola.append(definite_integral(0, 3) - parabola_method(0, 3, i))
        values_cubic.append(definite_integral(0, 3) - cubic_parabola_method(0, 3, i))
        values_boole.append(definite_integral(0, 3) - boole_method(0, 3, i))
        i -= 0.01
    fig, ax = plt.subplots()
    ax.plot(values_x, values_rectangle, label='rectangle')
    ax.plot(values_x, values_trapezoidal, label='trapezoidal')
    ax.plot(values_x, values_parabola, label='parabola')
    ax.plot(values_x, values_cubic, label='cubic')
    ax.plot(values_x, values_boole, label='boole')
    ax.invert_xaxis()
    ax.legend()
    ax.set(xlabel='x', ylabel='f', title='5 * (np.sin(x) - 3) ** 2 + 1')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    a_ = 0
    b_ = 3
    c_ = 0
    d_ = 3
    step_ = 1e-3

    draw_function()

    print('analytical integral value at a = {0} and b = {1} is {2}'
          .format(a_, b_, definite_integral(a_, b_)))
    print('numerical integral value at a = {0} and b = {1} with step = {2} is {3} (rectangle method)'
          .format(a_, b_, step_, rectangle_method(a_, b_, step_)))
    print('numerical integral value at a = {0} and b = {1} with step = {2} is {3} (trapezoidal method)'
          .format(a_, b_, step_, trapezoidal_method(a_, b_, step_)))
    print('numerical integral value at a = {0} and b = {1} with step = {2} is {3} (parabola method)'
          .format(a_, b_, step_, parabola_method(a_, b_, step_)))
    print('numerical integral value at a = {0} and b = {1} with step = {2} is {3} (cubic parabola method)'
          .format(a_, b_, step_, cubic_parabola_method(a_, b_, step_)))
    print('numerical integral value at a = {0} and b = {1} with step = {2} is {3} (boole method)'
          .format(a_, b_, step_, boole_method(a_, b_, step_)))
    print('numerical integral value at a = {0} and b = {1} is {2} (gauss method - 2)'
          .format(a_, b_, gauss_method(a_, b_, 2)))
    print('numerical integral value at a = {0} and b = {1} is {2} (gauss method - 3)'
          .format(a_, b_, gauss_method(a_, b_, 3)))
    print('numerical integral value at a = {0} and b = {1} is {2} (gauss method - 4)'
          .format(a_, b_, gauss_method(a_, b_, 4)))
    print('numerical integral value at a = {0} and b = {1} is {2} (gauss method - 5)'
          .format(a_, b_, gauss_method(a_, b_, 5)))
    print('numerical integral value at a = {0} and b = {1} is {2} (gauss method - 6)'
          .format(a_, b_, gauss_method(a_, b_, 6)))
    print('numerical integral value at a = {0} and b = {1} with n = {2} is {3} (monte carlo method)'
          .format(a_, b_, 100, monte_carlo_method(a_, b_, 100)))
    print('analytical integral value at a = {0} b = {1} c = {2} d = {3} is {4}'
          .format(a_, b_,  c_, d_, definite_integral_xy()))
    print('numerical integral value at a = {0} b = {1} c = {2} d = {3} with n = {4} is {5} (monte carlo method)'
          .format(a_, b_,  c_, d_, 100, monte_carlo_method_xy(a_, b_, c_, d_, 100)))
