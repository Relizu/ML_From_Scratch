import numpy as np

def MSE(x,y,model):
    return np.sum(np.square(y-model.Predict(x)))/len(y)