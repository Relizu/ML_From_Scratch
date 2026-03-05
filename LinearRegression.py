import numpy as np
import random

class LinearRegression():
    def __init__(self, N=1):
        self.w =np.random.randn(N)
        self.b =np.random.randn()
    def Loss(self,x,y):
        return np.sum(np.square(y-self.Predict(x)))/len(y)

    def Predict(self, x):
        if x.ndim == 1:
            return np.dot(self.w, x) + self.b  # dot product = single value
        else:
            return x @ self.w + self.b         # matrix mult for batch
        
    def Step(self,x,y, Stochastic= False):
        if Stochastic:
            rand = random.randint(0,len(x)-1)
            error = float(y[rand] - self.Predict(x[rand]))
            # 3x1     3x1           3x1         1
            self.w = self.w +0.01 * x[rand] *error
            self.b = self.b+ 0.01 * error
        else:
            error = y - self.Predict(x)
            # 3x1         3x1  +        3x10 . 10x1
            self.w = self.w+ 0.01 * (x.T@error)/len(y)
            self.b = self.b+ 0.01 * np.sum(error)/len(y)