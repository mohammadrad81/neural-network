import numpy as np


class Loss:
    def __init__(self):
        pass

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError()

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError()
