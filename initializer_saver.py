import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Define constants of the to be trained network
name = "ReLU_1D_D1_V6"
input_dim = 1
net_width = 16
data_dim = 1000000 #We take 100000 samples between 0 and 1 and compute the function output
min_data = 0.0
max_data = 1.0
depth = 1 #Nr of layers->integer
act_fun = "relu" #Not used -> activation layer leaky relu->convergence issues
optim = "sgd"
loss_function = "mse"
nr_of_epochs = 20

def app_func(input):
    return np.sin(2*np.pi*input)

for j in range(0, 300):
    #Setup of the network
    initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
    bias_init = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
    model = keras.Sequential()
    for i in range(0, depth):
        model.add(keras.layers.Dense(net_width, use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(input_dim,)))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(keras.layers.Dense(1, use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.compile(optimizer = optim, loss=loss_function)
    #model.summary()
    model.save("Models_init_1D/"+name+"_"+str(j+1)+'.h5')
    print(j)

#filepath = "Models_1D/"+name+"-model-{epoch:02d}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
#callbcak_list = [checkpoint]
