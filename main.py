import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp


def function(x):
    return x ** 4 * np.sin(x) + 1 / np.sqrt(x + 1)


def scipy_integral(a, b):
    return sp.quad(function, a, b)[0]


def function_xy(x, y):
    return y ** 4 * np.sin(x) + 1 / np.sqrt(x + 1)


def scipy_integral_xy(a, b, c, d):
    return sp.dblquad(function_xy, a, b, c, d)[0]


def rectangle_method(a, b, step):
    n = int((b - a) // step)
    remainder = (b - a) % step
    sum_ = 0
    for i in range(n):
        sum_ += function(a + i * step + step / 2)
    sum_ *= step
    sum_ += remainder * function(b - remainder / 2)
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
        return (b - a) / 2 * (0.2369 * function((a + b) / 2 - (b - a) / 2 * 0.9061)
                              + 0.4786 * function((a + b) / 2 - (b - a) / 2 * 0.5384)
                              + 0.5688 * function((a + b) / 2)
                              + 0.4786 * function((a + b) / 2 + (b - a) / 2 * 0.5384)
                              + 0.2369 * function((a + b) / 2 + (b - a) / 2 * 0.9061))
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


def rectangle_method_xy(a, b, c, d, step):
    n = int((b - a) // step)
    remainder_x = (b - a) % step
    k = int((d - c) // step)
    remainder_y = (d - c) % step
    sum_ = 0
    for j in range(k):
        for i in range(n):
            sum_ += function_xy(a + i * step + step / 2, c + j * step + step / 2)
    sum_ *= step ** 2
    sum_remainder_x = 0
    for j in range(k):
        sum_remainder_x += function_xy(b - remainder_x / 2, c + j * step + step / 2)
    sum_remainder_x *= step * remainder_x
    sum_ += sum_remainder_x
    sum_remainder_y = 0
    for i in range(n):
        sum_remainder_y += function_xy(a + i * step + step / 2, d - remainder_y / 2, )
    sum_remainder_y *= step * remainder_y
    sum_ += sum_remainder_y
    sum_ += remainder_x * remainder_y * (function_xy(b - remainder_x / 2, d - remainder_y / 2))
    return sum_


def draw_function(a, b):
    values_x = []
    values_rectangle = []
    values_trapezoidal = []
    values_parabola = []
    values_cubic = []
    values_boole = []
    i = 1
    while i > 1e-3:
        values_x.append(i)
        values_rectangle.append(scipy_integral(a, b) - rectangle_method(a, b, i))
        values_trapezoidal.append(scipy_integral(a, b) - trapezoidal_method(a, b, i))
        values_parabola.append(scipy_integral(a, b) - parabola_method(a, b, i))
        values_cubic.append(scipy_integral(a, b) - cubic_parabola_method(a, b, i))
        values_boole.append(scipy_integral(a, b) - boole_method(a, b, i))
        i -= 0.01
    fig, ax = plt.subplots()
    ax.plot(values_x, values_rectangle, label='rectangle')
    ax.plot(values_x, values_trapezoidal, label='trapezoidal')
    ax.plot(values_x, values_parabola, label='parabola')
    ax.plot(values_x, values_cubic, label='cubic')
    ax.plot(values_x, values_boole, label='boole')
    ax.invert_xaxis()
    ax.legend()
    ax.set(xlabel='step', ylabel='error', title='x ** 4 * np.sin(x) + 1 / np.sqrt(x + 1) error graph')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    a_ = 0
    b_ = 3
    c_ = 0
    d_ = 3
    step_ = 1e-3
    n_ = 1000
    step_xy = 1e-2

    draw_function(a_, b_)

    print('integral value at a = {0} and b = {1} is {2} (scipy integration)'
          .format(a_, b_, scipy_integral(a_, b_)))
    print('integral value at a = {0} and b = {1} with step = {2} is {3} (rectangle method)'
          .format(a_, b_, step_, rectangle_method(a_, b_, step_)))
    print('integral value at a = {0} and b = {1} with step = {2} is {3} (trapezoidal method)'
          .format(a_, b_, step_, trapezoidal_method(a_, b_, step_)))
    print('integral value at a = {0} and b = {1} with step = {2} is {3} (parabola method)'
          .format(a_, b_, step_, parabola_method(a_, b_, step_)))
    print('integral value at a = {0} and b = {1} with step = {2} is {3} (cubic parabola method)'
          .format(a_, b_, step_, cubic_parabola_method(a_, b_, step_)))
    print('integral value at a = {0} and b = {1} with step = {2} is {3} (boole method)'
          .format(a_, b_, step_, boole_method(a_, b_, step_)))
    print('integral value at a = {0} and b = {1} is {2} (gauss method - 2)'
          .format(a_, b_, gauss_method(a_, b_, 2)))
    print('integral value at a = {0} and b = {1} is {2} (gauss method - 3)'
          .format(a_, b_, gauss_method(a_, b_, 3)))
    print('integral value at a = {0} and b = {1} is {2} (gauss method - 4)'
          .format(a_, b_, gauss_method(a_, b_, 4)))
    print('integral value at a = {0} and b = {1} is {2} (gauss method - 5)'
          .format(a_, b_, gauss_method(a_, b_, 5)))
    print('integral value at a = {0} and b = {1} is {2} (gauss method - 6)'
          .format(a_, b_, gauss_method(a_, b_, 6)))
    print('integral value at a = {0} and b = {1} with n = {2} is {3} (monte carlo method)'
          .format(a_, b_, n_, monte_carlo_method(a_, b_, n_)))
    print('integral value at a = {0} b = {1} c = {2} d = {3} is {4} (scipy double integration)'
          .format(a_, b_, c_, d_, scipy_integral_xy(a_, b_, c_, d_)))
    print('integral value at a = {0} b = {1} c = {2} d = {3} with step = {4} is {5} (rectangle method)'
          .format(a_, b_, c_, d_, step_xy, rectangle_method_xy(a_, b_, c_, d_, step_xy)))
    print('integral value at a = {0} b = {1} c = {2} d = {3} with n = {4} is {5} (monte carlo method)'
          .format(a_, b_, c_, d_, n_, monte_carlo_method_xy(a_, b_, c_, d_, n_)))
