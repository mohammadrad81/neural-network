import os

from src.layers.convolution2d import Conv2D
from src.layers.fullyconnected import FC
from src.layers.maxpooling2d import MaxPool2D
from src.losses.binarycrossentropy import BinaryCrossEntropy
from src.losses.meansquarederror import MeanSquaredError
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
images = images/255
target = [([1] * len(twos)) + ([0] * len(fives))]
target = np.array(target).reshape(len(target[0]), 1)
# target = target.reshape(target.shape[0], 1, 1, 1)
print(target.shape)
#
fc_1 = FC(input_size=784, output_size=100, name='fc1')
fc_2 = FC(input_size=100, output_size=100, name='fc2')
fc_3 = FC(input_size=100, output_size=100, name='fc3')
fc_4 = FC(input_size=100, output_size=1, name='fc4')
activation = Tanh()
sigmoid = Sigmoid()
layers_list=[
    ('fc1', fc_1),
    ('activation1', activation),
    ('fc2', fc_2),
    ('activation2', activation),
    ('fc3', fc_3),
    ('activation3', activation),
    ('fc4', fc_4),
    ('sigmoid', activation),
]

model = Model(
    layers_list,
    criterion=BinaryCrossEntropy(),
    optimizer=GD(layers_list=layers_list, learning_rate=0.001)
)


# In[17]:


model.train(images, target, epochs=5000)
