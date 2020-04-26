import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###We start with independent buckets, 1D uniformly distributed
###The sanity check should be mutual information equals 0 and nr of regions according to formula
###we set biasses equal to 1? -> for now yes

###Constants
min_part = -1
max_part = 1
nr_of_buckets = 16 ###Will be different in the end->not a constant anymore!
width = 16
D = 1
visualisation = 0
percentage_of_correlation = 0.5 #between 0 and 1

###This number will simulate correlation by allowing overlap
nr_corr = int(np.round(nr_of_buckets - percentage_of_correlation*(nr_of_buckets-1)))
nr_of_buckets = nr_of_buckets + nr_corr - 1

### Define buckets! start equidistant partitioning of 1D between -1 and 1 in nr_of_buckets buckets.
buckets = np.linspace(min_part, max_part, nr_of_buckets)

###Expected number of regions 1D independent case!
if D == 1:
    expected_number = nr_of_buckets*(1-(1-1/nr_of_buckets)**width)

###Mutual information computation
##We first compute f_w
f_w = np.zeros(nr_of_buckets)
if percentage_of_correlation != 0:
    for i in range(0, width):
        for j in range(0, nr_corr):
            f_w[i+j] = f_w[i+j] + 1/nr_corr
    f_w = f_w*1/width
else:
    for i in range(0, nr_of_buckets):
        f_w[i] = 1/nr_of_buckets
print("sanity check for distribution: " + str(np.sum(f_w)))

###Compute H_w
H_w = 0
for i in range(0, nr_of_buckets):
    if f_w[i] != 0:
        H_w = H_w + f_w[i]*np.log2(1/f_w[i])
print("H_w: "+str(H_w))

###Compute H_w1,...,wN -> hard!



### We take an example
def reg_count(input):
    count = 16
    (unique, counts) = np.unique(input, return_counts=True)
    unique = unique - 1
    for i in counts:
        count = count - i + 1
    return count
#Define the network
model = keras.Sequential()
model.add(keras.layers.Dense(width, use_bias=True, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(D,)))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.Dense(1, use_bias=True, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.compile(optimizer = "sgd", loss="mse")

#weight_vector redifinition after our use
weights = model.get_weights()
for i in range(0, width):
    index = np.random.randint(low = 0, high = nr_of_buckets)
    weights[0][0][i] = buckets[index]
    weights[1][i] = 1
model.set_weights(weights)
model.compile(optimizer = "sgd", loss="mse")

nr_of_regions = reg_count(weights[0][0][:])

###Print results
print("nr_of_regions: "+str(nr_of_regions))
print("expected_regions: "+str(expected_number))

###Visualisation
if visualisation == 1:
    x = np.linspace(-10, 10, 500)
    y = model.predict(x)
    plt.figure(1)
    plt.plot(x, y)
    plt.show()