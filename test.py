import numpy as np

from DecisionTree import *

# X = [Outlook, Temperature, Humidity, Windy]
X = np.array([
    [0, 0, 0, 0],  # 1
    [0, 0, 0, 1],  # 2
    [1, 0, 0, 0],  # 3
    [2, 1, 0, 0],  # 4
    [2, 2, 1, 0],  # 5
    [2, 2, 1, 1],  # 6
    [1, 2, 1, 1],  # 7
    [0, 1, 0, 0],  # 8
    [0, 2, 1, 0],  # 9
    [2, 1, 1, 0],  # 10
    [0, 1, 1, 1],  # 11
    [1, 1, 0, 1],  # 12
    [1, 0, 1, 0],  # 13
    [2, 1, 0, 1],  # 14
    ])

y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
tree = DecisionTree(X,y,10)
print(tree.predict(np.array([0,1,0,0])))