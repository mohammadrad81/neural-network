#!/usr/bin/env python
# coding: utf-8

# # Testing model

# In[1]:


from src.activations import LinearActivation
from src.layers.fullyconnected import FC
from src.losses.meansquarederror import MeanSquaredError
from src.model import Model
from src.optimizers.gradientdescent import GD
from src.activations import ReLU, Sigmoid, Tanh


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv("datasets/california_houses_price/california_housing_train.csv")


# In[4]:


# data


# In[5]:


X_train, y_train = data.drop("median_house_value", axis=1), data[["median_house_value"]]


# In[6]:


# X_train


# In[7]:


y_train


# In[8]:


X_train_arr = np.array(X_train)


# In[9]:


y_train_arr = np.array(y_train)

y_train_arr = 10 * y_train_arr / np.max(y_train_arr)


# In[10]:

#
# X_train_arr
#
#
# # In[11]:
#
#
# y_train_arr
#
#
# # In[12]:
#
#
# X_train_arr.shape


# In[13]:


fc_1 = FC(input_size=8, output_size=100, name='fc1')
fc_2 = FC(input_size=100, output_size=100, name='fc2')
fc_3 = FC(input_size=100, output_size=100, name='fc3')
fc_4 = FC(input_size=100, output_size=1, name='fc4')

# In[14]:


linear = LinearActivation()
relu = ReLU()

# In[15]:


layers_list=[
    ('fc1', fc_1),
    ('linear', ReLU()),
    ('fc2', fc_2),
    ('linear', ReLU()),
    ('fc3', fc_3),
    ('linear', ReLU()),
    ('fc4', fc_4),
    ('linear', linear),
]


# In[16]:


model = Model(
    layers_list,
    criterion=MeanSquaredError(),
    optimizer=GD(layers_list=layers_list, learning_rate=0.1)
)


# In[17]:


model.train(X_train_arr, y_train_arr, epochs=10000)


# In[ ]:




