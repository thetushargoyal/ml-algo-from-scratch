import numpy as np

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, 
                 feature = None,
                 threshold = None,
                 left=None,
                 right=None,
                 *,
                 value=None):
        self.threshold = threshold
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value if not None
    
class DecisionTree:
    
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass