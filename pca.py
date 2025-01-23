import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components =n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eignevectors = np.linalg.eig(cov)
        eignevectors = eignevectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eignevectors[idxs]

        self.components = eignevectors[0:self.n_components]
        
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)