import numpy as np
from src.layers.layer import Layer


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                 initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.initialize_method = initialize_method

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.parameters = [self.initialize_weights(), self.initialize_bias()]

    def initialize_weights(self):
        """
        Initialize weights.
        returns:
            weights: initialized kernel with shape: (kernel_size[0], kernel_size[1], in_channels, out_channels)
        """
        # TODO: Implement initialization of weights

        if self.initialize_method == "random":
            return np.random.randn(self.kernel_size[0],
                                   self.kernel_size[1],
                                   self.in_channels,
                                   self.out_channels) * 0.01
        if self.initialize_method == "xavier":
            return None
        if self.initialize_method == "he":
            return None
        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        """
        Initialize bias.
        returns:
            bias: initialized bias with shape: (1, 1, 1, out_channels)
        
        """
        # TODO: Implement initialization of bias
        return np.zeros(shape=(1, 1, 1, self.out_channels))

    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the convolutional layer.
        args:
            input_shape: shape of the input to the convolutional layer
        returns:
            target_shape: shape of the output of the convolutional layer
        """
        # TODO: Implement calculation of target shape
        H = (input_shape[0] + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W = (input_shape[1] + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return (H, W)

    def pad(self, A, padding, pad_value=0):
        """
        Pad the input with zeros.
        args:
            A: input to be padded
            padding: tuple of padding for height and width
            pad_value: value to pad with
        returns:
            A_padded: padded input
        """
        A_padded = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant",
                          constant_values=(pad_value, pad_value))
        return A_padded

    def single_step_convolve(self, a_slic_prev, W, b):
        """
        Convolve a slice of the input with the kernel.
        args:
            a_slic_prev: slice of the input data
            W: kernel
            b: bias
        returns:
            Z: convolved value
        """
        # TODO: Implement single step convolution
        Z = W * a_slic_prev   # hint: element-wise multiplication
        Z = Z.sum()    # hint: sum over all elements
        Z += np.float64(b)    # hint: add bias as type float using np.float(None)
        return Z

    def forward(self, A_prev):
        """
        Forward pass for convolutional layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
            returns:
                A: output of the convolutional layer
        """
        # TODO: Implement forward pass
        Weights, bias = self.parameters
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, C) = self.kernel_size[0], self.kernel_size[1], self.out_channels
        stride_h, stride_w = self.stride[0], self.stride[1]
        padding_h, padding_w = self.padding[0], self.padding[1]
        H, W = self.target_shape(input_shape=(H_prev, W_prev))
        Z = np.zeros(shape=(batch_size, H, W, self.out_channels))
        A_prev_pad = self.pad(A_prev, self.padding)  # hint: use self.pad()
        for i in range(batch_size):
            for h in range(H):
                h_start = h * stride_h
                h_end = h_start + kernel_size_h
                for w in range(W):
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w
                    for c in range(self.out_channels):
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        Z[i, h, w, c] = self.single_step_convolve(a_slice_prev, Weights[:, :, :, c], bias[:, :, :, c])  # hint: use self.single_step_convolve()
        return Z

    def backward(self, dZ, A_prev):
        """
        Backward pass for convolutional layer.
        args:
            dZ: gradient of the cost with respect to the output of the convolutional layer
            A_prev: activations from previous layer (or input data)
            A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
        returns:
            dA_prev: gradient of the cost with respect to the input of the convolutional layer
            gradients: list of gradients with respect to the weights and bias
        """
        # TODO: Implement backward pass
        Weights, bias = self.parameters
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, C) = self.kernel_size[0], self.kernel_size[0], self.out_channels

        stride_h, stride_w = self.stride[0], self.stride[1]
        padding_h, padding_w = self.padding[0], self.padding[1]
        H, W = self.target_shape(input_shape=(H_prev, W_prev))
        dA_prev = np.zeros(A_prev.shape)  # hint: same shape as A_prev
        dW = np.zeros(Weights.shape)  # hint: same shape as W
        db = np.zeros(bias.shape)  # hint: same shape as b
        A_prev_pad = self.pad(A_prev, padding=self.padding)  # hint: use self.pad()
        dA_prev_pad = self.pad(A_prev, padding=self.padding)  # hint: use self.pad()
        for i in range(batch_size):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(H):
                h_start = h * stride_h
                h_end = h_start + kernel_size_h
                for w in range(W):
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w
                    for c in range(C):
                        a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]
                        W_helper = Weights[:, :, :, c]
                        dZ_helper = dZ[i, h, w, c]
                        da_prev_pad[h_start:h_end, w_start:w_end, :] += W_helper * dZ_helper  # hint: use element-wise multiplication of dZ and W
                        dW[..., c] += dZ[i, h, w, c] * a_slice  # hint: use element-wise multiplication of dZ and a_slice
                        db[..., c] += dZ[i, h, w, c]  # hint: use dZ
            if padding_h > 0 and padding_w > 0:
                dA_prev[i, :, :, :] = dA_prev_pad[i, padding_h:-padding_h, padding_w:-padding_w, :]  # hint: remove padding (trick: pad:-pad)
            elif padding_h > 0:
                dA_prev[i, :, :, :] = dA_prev_pad[i, padding_h:-padding_h, :, :]
            elif padding_w > 0:
                dA_prev[i, :, :, :] = dA_prev_pad[i, :, padding_w:-padding_w, :]
            else:
                dA_prev[i, :, :, :] = dA_prev_pad[i, :, :, :]
        grads = [dW, db]
        return dA_prev, grads

    def update(self, optimizer, grads):
        """
        Update parameters of the convolutional layer.
        args:
            optimizer: optimizer to use for updating parameters
            grads: list of gradients with respect to the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name, self.parameters)
