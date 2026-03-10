import numpy as np
import random

def GradientDescent(x,y,model):
    error = y - model.Predict(x)
    # 3x1         3x1  +        3x10 . 10x1
    model.w = model.w+ model.rate * (x.T@error)/len(y)
    model.b = model.b+ model.rate * np.sum(error)/len(y)

def StochasticGradientDescent(x,y,model):
    rand = random.randint(0,len(x)-1)
    error = float(y[rand] - model.Predict(x[rand]))
    # 3x1     3x1           3x1         1
    model.w = model.w + model.rate * x[rand] *error
    model.b = model.b + model.rate * error