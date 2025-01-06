# Bayes Theorem

# P(A|B) = (P(B|A)*P(A))/P(B)
# P(y|X) = (P(X|y)*P(y))/P(X)
# where X is a feature vector X = (x1, x2, x3...xn) 
# naive bayes -> independent features -> mutually independent

import numpy as np
from utils import accuracy
from sklearn.model_selection import train_test_split
from sklearn import datasets

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior = class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]


    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator
    
    def load_dataset(self):
        X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":

    nb = NaiveBayes()
    X_train, X_test, y_train, y_test = nb.load_dataset()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))