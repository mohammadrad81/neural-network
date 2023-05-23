import numpy as np
from .loss import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        pass

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        computes the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                mean squared error loss
        """
        # TODO: Implement mean squared error loss
        # batch_size = y_pred.shape[0]
        cost = np.mean(np.square(y_pred - y_true))
        # return np.squeeze(cost)
        return cost

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        computes the derivative of the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                derivative of the mean squared error loss
        """
        # TODO: Implement backward pass for mean squared error loss
        return 2 * (y_pred - y_true) / y_pred.shape[0]
