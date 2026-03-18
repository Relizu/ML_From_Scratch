import numpy as np
from NaiveBayes import *

# Vocabulary: ["great","movie","loved","terrible","film","awful","amazing","wonderful","best","boring","waste","horrible","bad","story","cast","disappointed"]
# Each row = word counts in that sentence

X_sentiment = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "great movie loved it"
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "terrible film awful acting"
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # "amazing wonderful best ever"
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # "boring waste of time"
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # "loved the story great cast"
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # "horrible bad disappointed"
])
y_sentiment = np.array([1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative

X_sentiment_test = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # "great story but boring ending"
])

model = NaiveBayes(X_sentiment,y_sentiment)
print(model.Predict(X_sentiment_test))