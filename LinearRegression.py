import numpy as np
import Loss
import Optimizers

class LinearRegression():
    def __init__(self, N=1):
        self.w =np.random.randn(N)
        self.b =np.random.randn()
        self.rate=0.01
    def Loss(self,x,y ,lossfunc = Loss.MSE):
        return lossfunc(x,y,self)

    def Predict(self, x):
        if x.ndim == 1:
            return np.dot(self.w, x) + self.b  # dot product = single value
        else:
            return x @ self.w + self.b         # matrix mult for batch
        
    def Step(self,x,y, Optimizer = Optimizers.GradientDescent):
        Optimizer(x,y,self)
