import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

amp = 1
frequency = 100
period = 1 / frequency
omega = 2 * np.pi * frequency


def signal(t):
    return amp * math.cos(omega * t)


def sig_cos(t, n):
    return signal(t) * math.cos(omega * n * t)


def sig_sin(t, n):
    return signal(t) * math.sin(omega * n * t)


def fourier(t, N=10):
    a0 = (2 / period) * quad(signal, 0, period)[0]

    res = (a0 / 2)
    for i in range(1, N + 1):
        an = (2 / period) * quad(sig_cos, 0, period, args=i)[0]
        bn = (2 / period) * quad(sig_sin, 0, period, args=i)[0]

        res += an * math.cos(omega * i * t) + bn * math.sin(omega * i * t)
    return res


x = np.linspace(0, 1, 1000)

y = amp * np.cos(x * omega)
y_app = np.array([fourier(t) for t in x])

y_fft = np.fft.ifft(y)

plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(x, y)
plt.title("Оригинальный сигнал")

plt.subplot(3, 1, 2)
plt.plot(x, y_app, color='darkorange')
plt.title("Аппроксимированная функция")

plt.subplot(3, 1, 3)
plt.plot(np.abs(y_fft[:len(y_fft) // 2]), 'm')
plt.title("Спектр сигнала")


plt.show()
