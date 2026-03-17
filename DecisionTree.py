import numpy as np

class DecisionTree:
    def __init__(self,x,y, depth =3 , index=0):
        self.x = x
        self.y = y
        self.depth = depth
        self.index = index
        self.right = None
        self.left = None
        self.best =[None,None,-1]
        if index < depth and len(np.unique(y)) > 1:
            parent_entropy = self.get_Entropy(self.y)
            for col in range(self.x.shape[1]):
                for item in np.unique(self.x[:, col]):
                    mask = self.x[:, col] == item
                    left, right = self.y[mask], self.y[~mask]
                    if len(left) == 0 or len(right) == 0:
                        continue
                    weighted_entropy = (len(left)/len(self.y))  * self.get_Entropy(left) \
                                    + (len(right)/len(self.y)) * self.get_Entropy(right)
                    ig = parent_entropy - weighted_entropy
                    if ig >  self.best[2]:
                         self.best = [col, item, self.get_Entropy(self.y)-weighted_entropy]
                    
            print( self.best)
            best_mask = self.x[:, self.best[0]] == self.best[1]
            self.right = DecisionTree(self.x[best_mask],self.y[best_mask],self.depth,self.index+1)
            self.left = DecisionTree(self.x[~best_mask],self.y[~best_mask],self.depth,self.index+1)
    
    def predict(self,x):
        current = self
        while current.right is not None or current.left is not None:
            if x[current.best[0]]== current.best[1]:
                current= current.right
            else:
                current = current.left
        return int(np.sum(current.y) > len(current.y) / 2)
    
    
    def get_Entropy(self, y):
        p = np.count_nonzero(y) / len(y)
        q = 1 - p
        return -p * np.log2(p + 1e-9) - q * np.log2(q + 1e-9)