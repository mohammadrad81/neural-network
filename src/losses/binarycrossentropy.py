import numpy as np
from src.losses.loss import Loss


class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        pass

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Computes the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                binary cross entropy loss
        """
        # TODO: Implement binary cross entropy loss
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                derivative of the binary cross entropy loss
        """
        # hint: use the np.divide function
        # TODO: Implement backward pass for binary cross entropy loss
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        d = -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred))) / len(y_pred)
        return d
