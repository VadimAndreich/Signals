import numpy as np
import matplotlib.pyplot as plt
import math


def taylor(x, degree):
    res = 0
    for k in range(0, degree // 2 + 1):
        res += ((-1) ** k) * ((x ** (2 * k + 1)) / (math.factorial(2 * k + 1)))
    return res


x = np.linspace(-1 * np.pi, np.pi, 100)
y = np.sin(x)

y_taylor_1 = taylor(x, 1)
y_taylor_3 = taylor(x, 3)
y_taylor_7 = taylor(x, 7)

plt.plot(x, y, label='sin(x)', color='black')
plt.plot(x, y_taylor_1, label='Тейлор 1 степень', color='green')
plt.plot(x, y_taylor_3, label='Тейлор 3 степень', color='blue')
plt.plot(x, y_taylor_7, label='Тейлор 7 степень', color='orange')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Разложение синуса в ряд Тейлора')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.legend()
plt.show()

