import os

from src.layers.convolution2d import Conv2D
from src.layers.fullyconnected import FC
from src.losses.binarycrossentropy import BinaryCrossEntropy
from src.optimizers.gradientdescent import GD
from src.activations import LinearActivation, Sigmoid, ReLU, Tanh
from src.model import Model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

print(os.listdir("datasets/MNIST/2/"))

list_of_2_images_file_names = os.listdir("datasets/MNIST/2/")[:10]
list_of_5_images_file_names = os.listdir("datasets/MNIST/5/")[:10]
fives = np.zeros(shape=(len(list_of_5_images_file_names), 28, 28, 1))
twos = np.zeros(shape=(len(list_of_2_images_file_names), 28, 28, 1))
for i in range(len(list_of_2_images_file_names)):
    image = plt.imread(f"datasets/MNIST/2/{list_of_2_images_file_names[i]}")
    twos[i, :, :, 0] = image
for i in range(len(list_of_5_images_file_names)):
    fives[i, :, :, 0] = plt.imread(f"datasets/MNIST/5/{list_of_5_images_file_names[i]}")
images = np.concatenate((twos, fives), axis=0)
target = [([1] * len(twos)) + ([0] * len(fives))]
target = np.array(target).reshape(len(target[0]), 1)
# target = target.reshape(target.shape[0], 1, 1, 1)
print(target.shape)
#
im: np.ndarray = plt.imread("datasets/MNIST/2/img_22.jpg")
im: np.ndarray = im.reshape(1, im.shape[0], im.shape[1], 1)
im = im.astype('float64')

# print(type(im))
# plt.imshow(im)
# plt.show()

cnn1 = Conv2D(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), name='cnn1')
activation = Tanh()
cnn2 = Conv2D(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), name='cnn2')
fully_connected1 = FC(input_size=20 * 20 * 1 * 1, output_size=100, name='fc1')
fully_connected2 = FC(input_size=100, output_size=1, name='fc2')
sigmoid = Sigmoid()
layers_list = [
    ("cnn1", cnn1),
    ("activation1", activation),
    ("cnn2", cnn2),
    ("activation2", activation),
    ('fc1', fully_connected1),
    ('activation3', activation),
    ('fc2', fully_connected2),
    ("sigmoid", sigmoid)
]

model = Model(
    layers_list=layers_list,
    criterion=BinaryCrossEntropy(),
    optimizer=GD(
        layers_list=layers_list,
        learning_rate=0.01 #6.931471807541806
    )
)

model.train(X=images, y=target, epochs=1000)
model.save(name="cnn.pkl")
print(model.predict(images[0].reshape(1, *images[0].shape)))
