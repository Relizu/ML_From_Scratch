import numpy as np
from KNN import *

# Features: [age, income (k), hours_online_per_day]
# Labels:   0 = budget shopper, 1 = regular, 2 = premium

X = np.array([
    # Budget shoppers (0)
    [22, 18, 1.5],
    [19, 12, 2.0],
    [25, 22, 1.0],
    [30, 25, 0.5],
    [21, 15, 3.0],
    [28, 20, 1.2],
    [24, 17, 2.5],
    [20, 13, 1.8],

    # Regular shoppers (1)
    [35, 55, 3.5],
    [40, 60, 2.5],
    [33, 50, 4.0],
    [38, 65, 3.0],
    [45, 70, 2.0],
    [32, 48, 3.8],
    [42, 62, 2.8],
    [37, 58, 3.2],

    # Premium shoppers (2)
    [50, 120, 5.0],
    [55, 150, 4.5],
    [48, 110, 6.0],
    [60, 180, 3.5],
    [52, 130, 5.5],
    [58, 160, 4.0],
    [47, 105, 5.8],
    [53, 140, 4.8],
])

y = np.array([0,0,0,0,0,0,0,0,
     1,1,1,1,1,1,1,1,
     2,2,2,2,2,2,2,2])


test_point = np.array([20, 57, 3.3]) 

model = KNN(X,y,5)

print(model.Predict(test_point))