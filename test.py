import numpy as np
import time

from LogisticRegression import *

a = np.array([
    [1, 4, 0],
    [2, 5, 1],
    [1, 6, 0],
    [3, 4, 1],
    [4, 7, 2],
    [5, 6, 2],
    [6, 8, 3],
    [7, 5, 3],
    [8, 7, 4],
    [9, 8, 5],
])


b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(3)
start = time.time()

for _ in range(1000000):
    model.Step(a, b, Optimizers.GradientDescent)
    if _ % 500 == 0:
        print(f"Epoch {_} | Loss: {model.Loss(a,b,Loss.BCE)}")
print(time.time()-start)
print(model.w,model.b)
print(model.Evaluate(a))
print(b)