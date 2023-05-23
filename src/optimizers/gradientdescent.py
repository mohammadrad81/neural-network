# TODO: Implement the gradient descent optimizer
from typing import List, Tuple, Dict
from src.layers.layer import Layer
class GD:
    def __init__(self, layers_list: List[Tuple[str, Layer]], learning_rate: float):
        """
        Gradient Descent optimizer.
            args:
                layers_list: dictionary of layers name and layer object
                learning_rate: learning rate
        """
        self.learning_rate: float = learning_rate
        self.layers_list: List[Tuple[str, Layer]] = layers_list
        self.layers_dict: Dict[str, Layer] = {name: layer for (name, layer) in self.layers_list}
    
    def update(self, grads, name, previous_params):
        """
        Update the parameters of the layer.
            args:
                grads: list of gradients for the weights and bias
                name: name of the layer
            returns:
                params: list of updated parameters
        """
        params = []
        # print(f"previous params: {previous_params}")
        # print(f"grads[{name}]: {grads[name]}")

        #TODO: Implement gradient descent update
        for i in range(len(grads[name])):
            params.append(previous_params[i] - self.learning_rate * grads[name][i])
        # print(f"new_params: {params}")
        return params