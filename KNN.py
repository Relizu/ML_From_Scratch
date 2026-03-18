import numpy as np

class KNN:

    def __init__(self,x,y,k=3):
        self.x = x
        self.y = y
        self.k = k
    
    def Predict(self,x):
        
        distances = np.linalg.norm(self.x - x, axis=1)
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_labels = self.y[k_indices]
        
        return np.bincount(k_labels).argmax()