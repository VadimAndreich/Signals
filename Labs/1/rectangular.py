import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


amp = 2
period = 2
frequency = 1 / 2
omega = 2 * np.pi * frequency


def signal(t):
    value = t % period
    if value <= period / 2:
        return amp
    else:
        return -amp


def sig_cos(t, n):
    return signal(t) * math.cos(omega * n * t)


def sig_sin(t, n):
    return signal(t) * math.sin(omega * n * t)


def fourier(t, N=10):
    a0 = (2 / period) * quad(signal, 0, period)[0]

    res = (a0 / 2)
    for n in range(1, N + 1):
        an = (2 / period) * quad(sig_cos, 0, period, args=n)[0]
        bn = (2 / period) * quad(sig_sin, 0, period, args=n)[0]

        res += an * math.cos(omega * n * t) + bn * math.sin(omega * n * t)
    return res


x = np.linspace(-4, 4, 1000)
y = np.array([signal(t) for t in x])

y_app = np.array([fourier(t) for t in x])
y_err = y_app - y

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)

plt.plot(x, y)
plt.plot(x, y_app)
plt.title("График сигнала и аппроксимированная функция")

plt.subplot(2, 1, 2)
plt.plot(x, y_err, 'm')
plt.title("Ошибка аппроксимирования")

plt.show()
