import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Constants
filepath_base = "Models_1DSIN/LReLU_1D_D1_SINE_V"
#filepath_base = "Models_1D/ReLU_1D_D2_V"
filepath_middle = "-model-"
filepath_ending = ".hdf5"
epoch_nr = 50
nr_of_versions = 5

#Structure that will define the preciseness of the distribution.
nr_of_grids = 52 #For now make this even please!
width = 0.2 #Size of the buckets
nr_of_grids_bias = 30
width_bias = 0.2
center_w = 0.0 #Center point for the grid of the weights
center_b = 0.0 #Center point for the grid of the biasses


### Open the models
models = []
###First number = number of same experiment. Second number = epoch
if epoch_nr < 10:
    epoch = "0"+str(epoch_nr)
else:
    epoch = str(epoch_nr)
for i in range(1, nr_of_versions+1):
    models.append(keras.models.load_model(filepath_base+str(i)+filepath_middle+epoch+filepath_ending))
models = np.array(models)

### Distibution: start->positive-negative, multiplicative and adding term seen as equivalent
distr = np.zeros((nr_of_grids, nr_of_versions))
biasses = np.zeros((nr_of_grids_bias, nr_of_versions))
nr_b = nr_of_grids/2 #We need the number of buckets/2!
nr_b_bias = nr_of_grids_bias/2
for k in range(0, nr_of_versions):
    model = models[k]

    ### extract the weights
    weights = np.array(model.get_weights())
    for i in range(0, len(weights)-1): #last one is the bias
        m = weights[i]
        if i == 0:
            m = m[0]
        if i%2 == 0:
            for j in range(0, len(m)):
                if i == 0 or i == len(weights)-1:
                    bucket_nr = int(np.floor((m[j]-center_w)/width)+nr_b)
                    if bucket_nr < 0:
                        distr[0][k] = distr[0][k]+1
                    elif bucket_nr >= nr_of_grids:
                        distr[nr_of_grids-1][k] = distr[nr_of_grids-1][k] + 1
                    else:
                        distr[bucket_nr][k] = distr[bucket_nr][k] + 1
                else:
                    ww = m[j]
                    for ll in range(0, len(ww)):
                        bucket_nr = int(np.floor((ww[ll]-center_w)/width)+nr_b)
                        if bucket_nr < 0:
                            distr[0][k] = distr[0][k]+1
                        elif bucket_nr >= nr_of_grids:
                            distr[nr_of_grids-1][k] = distr[nr_of_grids-1][k] + 1
                        else:
                            distr[bucket_nr][k] = distr[bucket_nr][k] + 1
        else:
            for j in range(0, len(m)):
                bucket_nr = int(np.floor((m[j]-center_b)/width_bias)+nr_b_bias)
                if bucket_nr < 0:
                    biasses[0][k] = biasses[0][k]+1
                elif bucket_nr >= nr_of_grids_bias:
                    biasses[nr_of_grids_bias-1][k] = biasses[nr_of_grids_bias-1][k] + 1
                else:
                    biasses[bucket_nr][k] = biasses[bucket_nr][k] + 1

def mean_array(input_a, grids):
    mean = np.zeros(grids)
    for it in range(0, grids):
        for i in range(0, nr_of_versions):
            mean[it] = mean[it] + input_a[it][i]
    mean = 1/nr_of_versions*mean
    return mean

def std_var_array(input_a, means, grids):
    variance = np.zeros(grids)
    for it in range(0, grids):
        for i in range(0, nr_of_versions):
            variance[it] = variance[it] + (means[it] - input_a[it][i])*(means[it] - input_a[it][i])
    variance = variance/(nr_of_versions-1)
    std_dev = np.sqrt(variance)
    return std_dev

mean_w = mean_array(distr, nr_of_grids)
mean_b = mean_array(biasses, nr_of_grids_bias)
std_dev_w = std_var_array(distr, mean_w, nr_of_grids)
std_dev_b = std_var_array(biasses, mean_b, nr_of_grids_bias)

print("means_w: "+str(mean_w))
print("means_b: "+str(mean_b))
print("std_dev_w: "+str(std_dev_w))
print("std_dev_b: "+str(std_dev_b))

###We plot the results
plt.figure(1)
x_axis = (np.array(range(0,nr_of_grids))+0.5-nr_of_grids/2)*width + center_w
plt.bar(x_axis, mean_w, width = 0.5*width, linewidth = 0.5, yerr = std_dev_w)
plt.title("Histogram weights of "+filepath_base+"-"+str(epoch_nr))
plt.xlabel("Weight values")
plt.ylabel("Mean of the counts")

plt.figure(2)
x_axis = (np.array(range(0,nr_of_grids_bias))+0.5-nr_of_grids_bias/2)*width_bias + center_b
plt.bar(x_axis, mean_b, width = 0.5*width_bias, linewidth = 0.5, yerr = std_dev_b)
plt.title("Histogram biasses of "+filepath_base+"-"+str(epoch_nr))
plt.xlabel("Weight values")
plt.ylabel("Mean of the counts")

plt.show()
####Format Depth 1 case of the weights
# 0:    weights w1 of input to hidden layer
# 1:    adding term b of input to hidden layer
# 2:    weights w2 of hidden layer to output
# 3:    bias component output

### Counting linear regions 1 dimention
