import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def app_func(input, d):
    #print(input)
    y = np.ones(10000)
    for i in range(0, d):
        y = y*np.transpose(input[i][:])
    return np.sin(2*np.pi*y)

N = 100
for D in range(0, 1):
    d = 2**D
    filepath_pre = "Dim_Experiment/D"+str(d)+"/"
    for i in range(0, N):
        name = str(i)
        input_dim = d
        net_width = 16
        data_dim = 10000 #We take 10000 samples between 0 and 1 and compute the function output
        min_data = 0.0
        max_data = 1.0
        depth = 1 #Nr of layers->integer
        act_fun = "relu" #Not used -> activation layer leaky relu->convergence issues
        initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
        bias_init = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
        optim = "sgd"
        loss_function = "mse"
        nr_of_epochs = 3


        model = keras.Sequential()
        for j in range(0, depth):
            model.add(keras.layers.Dense(net_width, use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(input_dim,)))
            model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(keras.layers.Dense(1, use_bias=True, kernel_initializer=initializer, bias_initializer=bias_init, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        model.compile(optimizer = optim, loss=loss_function)
        model.summary()

        filepath = filepath_pre+name+"-model-{epoch:02d}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
        callbcak_list = [checkpoint]

        #Generate data
        X_data = np.random.uniform(low= min_data, high = max_data, size = (data_dim, d))
        Y_data = app_func(np.transpose(X_data), d)

        #Train the network
        model.fit(x=X_data, y=Y_data, batch_size=None, epochs=nr_of_epochs, verbose=1, callbacks=callbcak_list, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

        #save the network
        model.save("SQModels_1D_D1_W16_3/"+name+'.h5') ##Use load_model(file_path) to get weights and model back
