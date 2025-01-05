import numpy as np 

def eudlidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)