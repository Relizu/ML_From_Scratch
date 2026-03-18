import numpy as np

class NaiveBayes:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.classes = np.unique(y)
        self.prob_arr = []

        for c in self.classes:
            prior = np.sum(y == c) / len(y)
            likelihood = self.prob(x[y == c])
            self.prob_arr.append((prior, likelihood))

    def prob(self, x):
        n_samples = x.shape[0]
        probs = []

        for i in range(x.shape[1]):
            p = (np.sum(x[:, i]) + 1) / (n_samples + 2)
            probs.append(p)

        return np.array(probs)

    def _predict_single(self, x):
        scores = []

        for prior, likelihood in self.prob_arr:
            log_score = np.log(prior) + np.sum(
                np.log(likelihood) * x + np.log(1 - likelihood) * (1 - x)
            )
            scores.append(log_score)

        return self.classes[np.argmax(scores)]

    def Predict(self, x):
        if x.ndim == 1:
            return self._predict_single(x)
        return np.array([self._predict_single(row) for row in x])