import numpy as np
import time

from LinearRegression import *

a = np.array([[1,2,3],
              [2,3,4],
              [3,1,2],
              [4,5,1],
              [5,2,6],
              [6,3,2],
              [7,4,5],
              [8,1,3],
              [9,6,2],
              [10,2,4]])
b = np.array([12,19,17,28,31,33,42,38,51,49])

model = LinearRegression(3)
start = time.time()
for _ in range(10000):
    model.Step(a, b, Stochastic=True)
    if _ % 500 == 0:
        print(f"Epoch {_} | Loss: {model.Loss(a,b)}")
print(time.time()-start)
print(model.w,model.b)