import numpy as np
from scipy.integrate import nquad
import math


def f(x, y, z):
    return (x + y + z) / math.sqrt(2 * x ** 2 + 4 * y ** 2 + 5 * z ** 2)


def xbounds():
    return [0, 1]


def ybounds(x):
    return [0, math.sqrt(1 - x ** 2)]


def zbounds(x, y):
    return [0, math.sqrt(1 - x ** 2 - y ** 2)]


I = nquad(lambda z, y, x: f(x, y, z), [zbounds, ybounds, xbounds])
print(I)
