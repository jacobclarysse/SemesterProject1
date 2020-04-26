import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Constants
filepath_base = "Models_1DSIN/LReLU_1D_D4_SINE_V4-model-"
#filepath_base = "Models_1D/ReLU_1D_D1_V4-model-"
filepath_ending = ".hdf5"
nr_of_versions = 5
epochs = 50
depth = 4


#Which weight/bias
layer_nr = 3
width_nr = 7
cross_width_nr = 6 #If applicable
bias = False #Multiplicative term or additive term?

#Structure that will define the preciseness of the distribution, centered around 0.
nr_of_grids = 20 #For now make this even please!
width = 0.001 #Size of the buckets
center = 0.14579291462898256 #Change the center of the distribution -> not necesarily 0 anymore, usefull to zoome in!

###open the needed models
ep_s = "01"
models= [keras.models.load_model(filepath_base+ep_s+filepath_ending)]
for i in range(2, epochs+1):
    ep_s = str(i)
    if i < 10:
        ep_s = "0"+ep_s
    models.append(keras.models.load_model(filepath_base+ep_s+filepath_ending))
models = np.array(models) #easy iterating

###Get the respective weight out each model and store in distr array in the right bucket
nr_b = nr_of_grids/2
distr = np.zeros(nr_of_grids)
mean = 0 
var_around_center = 0
for i in range(0,epochs):
    weights = np.array(models[i].get_weights())
    if layer_nr == 1 or layer_nr == depth+1:
        if layer_nr == 1 and bias == False:
            weights = weights[0]
        weight = weights[(layer_nr-1)*2+bias][width_nr] - center
        mean = mean + weight + center
        var_around_center = var_around_center + weight**2
        bucket_nr = int(np.floor(weight/width)+nr_b)
        if bucket_nr < 0:
            distr[0] = distr[0]+1
        elif bucket_nr >= nr_of_grids:
            distr[nr_of_grids-1] = distr[nr_of_grids-1] + 1
        else:
            distr[bucket_nr] = distr[bucket_nr] + 1
    else:
        weight = float(weights[(layer_nr-1)*2+bias][width_nr][cross_width_nr]) - center
        mean = mean + weight + center
        var_around_center = var_around_center + weight**2
        bucket_nr = int(np.floor(weight/width)+nr_b)
        if bucket_nr < 0:
            distr[0] = distr[0]+1
        elif bucket_nr >= nr_of_grids:
            distr[nr_of_grids-1] = distr[nr_of_grids-1] + 1
        else:
            distr[bucket_nr] = distr[bucket_nr] + 1

###Plot results
plt.figure(1)
x_axis = (np.array(range(0,nr_of_grids))+0.5-nr_of_grids/2)*width + center
print(distr)
print("mean: "+str(mean/epochs))
print("standard deviation around center: "+str(var_around_center/(epochs-1)))
plt.bar(x_axis, distr, width = 0.5*width)
plt.title("Weight tracking: example")
plt.xlabel("Weight value")
plt.ylabel("counts")
plt.show()
