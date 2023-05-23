from src.layers.convolution2d import Conv2D
from src.layers.maxpooling2d import MaxPool2D
from src.layers.fullyconnected import FC

from src.activations import Activation
from src.layers.layer import Layer
from src.losses.loss import Loss
import pickle
import tqdm
import numpy as np
from typing import Tuple, Dict, List
from src.losses.binarycrossentropy import BinaryCrossEntropy
from src.losses.meansquarederror import MeanSquaredError
from src.optimizers.gradientdescent import GD
from src.optimizers.adam import Adam

class Model:
    def __init__(self,
                 layers_list: List[Tuple[str, MaxPool2D | FC | Conv2D]],
                 criterion: MeanSquaredError | BinaryCrossEntropy,
                 optimizer: Adam | GD,
                 name: str=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model: Dict[str, Layer] = {l[0]: l[1] for l in layers_list}
            self.criterion: Loss = criterion
            self.optimizer: Adam | GD = optimizer
            self.layers_names = [l[0] for l in layers_list]
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)
    
    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        # TODO: Implement check if the layer is a layer
        return isinstance(layer, Layer)

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        # TODO: Implement check if the layer is an activation
        return isinstance(layer, Activation)
    
    def forward(self, x: np.ndarray):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        A = x
        tmp = [x]
        # TODO: Implement forward pass through the model
        # NOTICE: we have a pattern of layers and activations
        for l in range(0, len(self.layers_names), 2):
            Z = self.model[self.layers_names[l]].forward(A)
            tmp.append(Z)    # hint add a copy of Z to tmp
            A = self.model[self.layers_names[l + 1]].forward(Z)
            tmp.append(A)    # hint add a copy of A to tmp
        return tmp
    
    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        # TODO: Implement backward pass through the model
        # NOTICE: we have a pattern of layers and activations
        # for from the end to the beginning of the tmp list
        # print(f"len of tmp: {len(tmp)}")
        for l in range(len(tmp), 2, -2):
            # print('in for of backward')
            A_prev, Z, A = tmp[l - 3], tmp[l - 2], tmp[l - 1]
            activation_layer: Activation = self.model[self.layers_names[l - 2]]
            dZ = activation_layer.backward(dA, Z)
            layer: Layer = self.model[self.layers_names[l - 3]]
            dA, grad = layer.backward(dZ, A_prev)
            grads[self.layers_names[l - 3]] = grad
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for name in grads.keys():
            layer = self.model[name]
            # print(f'layer_name: {name}')
            if isinstance(layer, Layer) and not isinstance(layer, MaxPool2D):    # hint check if the layer is a layer and also is not a maxpooling layer
                # print("in model update if passed")
                self.model[name].update(self.optimizer, grads)
    
    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        # TODO: Implement one epoch of training
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion.compute(AL, y)
        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss
    
    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)
        
    def load_model(self, name):
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)
        
    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            return np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        # TODO: Implement batch
        # last_index =    # hint last index of the batch check for the last batch
        batch = order[batch_size * index: batch_size * (index + 1)]
        # NOTICE: inputs are 4 dimensional or 2 demensional
        # if len(X.shape) == 2:
        bx = X[batch]
        by = y[batch]
        # else: # X is 4 dimensional
        #     bx = None
        #     by = None
        return bx, by

    def compute_loss(self, X, y, batch_size): #???
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        # TODO: Implement compute loss
        m = X.shape[0]
        order = None
        cost = 0
        for b in range(m // batch_size):
            bx, by = None
            tmp = None
            AL = None
            cost += None
        return cost

    def train(self, X, y, epochs, val=None, batch_size=1000, shuffling=False, verbose=1, save_after=None):
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        # TODO: Implement training
        train_cost = []
        val_cost = []
        # NOTICE: if your inputs are 4 dimensional m = X.shape[0] else m = X.shape[1]
        m = X.shape[0]
        # print(f"before epoch for")
        for e in range(1, epochs + 1):
            # order = self.shuffle(m, shuffling)
            # cost = 0
            # for b in range(m // batch_size):
            #     bx, by = self.batch(X, y, batch_size, b, order)
            #     print(f'batch number: {b}')
            #     cost += self.one_epoch(bx, by)
            cost = self.one_epoch(X, y)
            train_cost.append(cost)
            if val is not None:
                val_cost.append(self.forward(X)[-1])
            if verbose != False:
                if e % verbose == 0:
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost
    
    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        # TODO: Implement prediction
        return self.forward(X)[-1]