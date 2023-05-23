class Layer:
    def __init__(self):
        pass

    def forward(self, X):
        raise NotImplementedError()

    def backward(self, dZ, A_prev):
        raise NotImplementedError()