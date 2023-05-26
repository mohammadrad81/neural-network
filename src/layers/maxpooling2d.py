import numpy as np
from src.layers.layer import Layer


class MaxPool2D(Layer):
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
        """
        Max pooling layer.
            args:
                kernel_size: size of the kernel
                stride: stride of the kernel
                mode: max or average
            Question:Why we don't need to set name for the layer?
        """
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode

    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the convolutional layer.
        args:
            input_shape: shape of the input to the convolutional layer
        returns:
            target_shape: shape of the output of the convolutional layer
        """
        # TODO: Implement calculation of target shape
        H = (input_shape[1] - self.kernel_size[0]) // self.stride[0] + 1
        W = (input_shape[2] - self.kernel_size[1]) // self.stride[1] + 1
        return (H, W)

    def forward(self, A_prev):
        """
        Forward pass for max pooling layer.
            args:
                A_prev: activations from previous layer (or input data)
            returns:
                A: output of the max pooling layer
        """
        # TODO: Implement forward pass for max pooling layer
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (f_h, f_w) = self.kernel_size[0], self.kernel_size[1]
        stride_h, stride_w = self.stride[0], self.stride[1]
        H, W = self.target_shape(A_prev.shape)
        A = np.zeros((batch_size, H, W, C_prev))
        for i in range(batch_size):
            for h in range(H):
                h_start = h * stride_h
                h_end = h_start + f_h
                for w in range(H):
                    w_start = w * stride_w
                    w_end = w_start + f_w
                    for c in range(C_prev):
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.average(a_prev_slice)
                        else:
                            raise ValueError("Invalid mode")

        return A

    def create_mask_from_window(self, x):
        """
        Create a mask from an input matrix x, to identify the max entry of x.
            args:
                x: numpy array
            returns:
                mask: numpy array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        # TODO: Implement create_mask_from_window
        mask = x == np.max(x)
        return mask

    def distribute_value(self, dz, shape):
        """
        Distribute the input value in the matrix of dimension shape.
            args:
                dz: input scalar
                shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
            returns:
                a: distributed value
        """
        # TODO: Implement distribute_value
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones(shape) * average
        return a

    def backward(self, dZ, A_prev):
        """
        Backward pass for max pooling layer.
            args:
                dA: gradient of cost with respect to the output of the max pooling layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: gradient of cost with respect to the input of the max pooling layer
        """
        # TODO: Implement backward pass for max pooling layer
        (f_h, f_w) = self.kernel_size
        strideh, stridew = self.stride
        batch_size, H_prev, W_prev, C_prev = A_prev.shape
        batch_size, H, W, C = dZ.shape
        dA_prev = np.zeros((batch_size, H_prev, W_prev, C_prev))
        for i in range(batch_size):
            for h in range(H_prev):
                for w in range(W_prev):
                    for c in range(C):
                        h_start = h * strideh
                        h_end = h_start + f_h
                        w_start = w * stridew
                        w_end = w_start + f_w
                        if self.mode == "max":
                            a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += np.multiply(mask, a_prev_slice)
                        elif self.mode == "average":
                            dz = dZ[i, h, w, c]
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += \
                                self.distribute_value(dz, (h_end-h_start, w_end-w_start))
                        else:
                            raise ValueError("Invalid mode")
        # Don't change the return
        return dA_prev, None
