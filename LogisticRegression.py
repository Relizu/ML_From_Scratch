import numpy as np
import Loss
import Activation
import Optimizers

class LogisticRegression:
    def __init__(self,N=1):
        self.w =np.random.randn(N)
        self.b =np.random.randn()
        self.rate=0.01
    
    def Loss(self,x,y ,lossfunc = Loss.BCE):
        return lossfunc(x,y,self)

    def Predict(self, x):
        if x.ndim == 1:
            return Activation.Sigmoid(np.dot(self.w, x) + self.b)
        else:
            return Activation.Sigmoid(x @ self.w + self.b)
        
    def Evaluate(self, x, threshold=0.5):
        if x.ndim == 1:
            return (Activation.Sigmoid(np.dot(self.w, x) + self.b) > threshold).astype(int)
        else:
            return (Activation.Sigmoid(x @ self.w + self.b) > threshold).astype(int)
        
    def Step(self,x,y, Optimizer = Optimizers.GradientDescent):
        Optimizer(x,y,self)
