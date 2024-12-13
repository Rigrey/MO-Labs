import numpy as np
from itertools import product

def method_execute(c, A, b, f, minimize):
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)

    upper_bounds = np.ceil(b / np.min(A + (A == 0), axis=0)).astype(int)
    x_ranges = [range(0, ub + 1) for ub in upper_bounds]

    max_F = float('-inf')
    best_x = None

    print("For c:", c)
    for x in product(*x_ranges):
        x = np.array(x)
        if np.all(np.dot(A, x) <= b):
            F = np.dot(c, x)
            print(x, " = ", F)
            if F > max_F:
                max_F = F
                best_x = x

    return [int(F), best_x.tolist()]