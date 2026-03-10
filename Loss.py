import numpy as np

def MSE(x,y,model):
    return np.sum(np.square(y-model.Predict(x)))/len(y)

def BCE(x,y,model):
    #     ylog(yy)+(1-y)log(1-yy)
    p = model.Predict(x)
    eps = 1e-9
    return -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))