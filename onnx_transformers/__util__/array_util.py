import numpy as np

def sigmoid(z: np.array):
    return 1 / (1 + np.exp(-z))