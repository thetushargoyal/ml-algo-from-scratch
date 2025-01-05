import numpy as np
from utils import eudlidean_distance
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        #compute distances
        distances = [eudlidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #get majority class
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def load_data(test_size = 0.2):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1234
        )
        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    
    k = 3
    knn = KNN(k=5)
    X_train, X_test, y_train, y_test = knn.load_data()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("KNN classification accuracy", knn.accuracy(y_test, predictions))