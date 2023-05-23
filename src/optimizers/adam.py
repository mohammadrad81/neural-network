import numpy as np
from .gradientdescent import GD


# TODO: Implement Adam optimizer
class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for layer_name in layers_list:
            # TODO: Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            v = [0 for p in layers_list[layer_name].parameters]
            s = [0 for p in layers_list[layer_name].parameters]
            self.V[layer_name] = v
            self.S[layer_name] = s

    def update(self, grads, name, epoch):
        layer = self.layers[name]
        params = []
        # TODO: Implement Adam update
        for i in range(len(grads)):
            self.V[name][i] = self.beta1 * self.V[name][i] + (1 - self.beta1) * grads[i]
            self.S[name][i] = self.beta2 * self.S[name][i] + (1 - self.beta2) * np.square(grads[i])
            self.V[name][i] /= (1 - np.power(self.beta1, epoch))  # TODO: correct V
            self.S[name][i] /= (1 - np.power(self.beta2, epoch))  # TODO: correct S
            params.append(grads[i] - self.learning_rate * self.V[name][i] / (np.sqrt(self.S[name][i]) + self.epsilon))
        return params