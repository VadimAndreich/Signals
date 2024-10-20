import numpy as np
from scipy.integrate import quad
import math


def f(x):
    return 2 if (0 <= x <= 1) else 0


I = quad(f, -1, 2)
print(I)