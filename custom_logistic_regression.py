import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=0.1, epochs=20000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        m, n = X.shape
        self.weights_ = np.zeros(n)
        self.bias_ = 0

        for i in range(self.epochs):
            z = np.dot(X, self.weights_) + self.bias_
            predictions = self.sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

        return self

    def predict(self, X):
        X = np.asarray(X)
        z = np.dot(X, self.weights_) + self.bias_
        return (self.sigmoid(z) >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        z = np.dot(X, self.weights_) + self.bias_
        probs = self.sigmoid(z)
        return np.c_[1 - probs, probs]
