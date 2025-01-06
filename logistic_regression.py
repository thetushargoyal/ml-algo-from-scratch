import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from utils import accuracy
## Some Notes from my reference

# not continous output but probability
# y_hat = 1 / (1 + e^(-2*x+b)) sigmoid function
# apply gradient descent
# cost function -> cross entropy

# update rules
# w = w - lr*dw
# b = b - lr*db



class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr * dw
            self.bias -= self.bias * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def load_dataset(self):

        bc = datasets.load_breast_cancer()
        X, y = bc.data, bc.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":

    regressor = LogisticRegression(lr=0.0001, n_iters=1000)
    X_train, X_test, y_train, y_test = regressor.load_dataset()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    print("LR classification accuracy:", accuracy(y_test, predictions))