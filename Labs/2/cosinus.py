import numpy as np
import matplotlib.pyplot as plt
import time


amp = 1
frequency1 = 50
frequency2 = 150
omega1 = 2 * np.pi * frequency1
omega2 = 2 * np.pi * frequency2


def signal(t):
    return amp * (np.cos(t * omega1) + np.cos(t * omega2))


def dft_slow(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


x = np.linspace(0, 1, 1000)
y = signal(x)

start = float(time.time())
np.fft.fft(y)
end = float(time.time())
print(end - start)

plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.show()
