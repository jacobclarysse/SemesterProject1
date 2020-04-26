import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### constants
filepath = "Models_2D/ReLU_2D_D1_V1-model-01.hdf5"
Dim = 2  #Dimention of the input space
width = 16

#We load the model and extract the weights -> Need example network!
model = keras.models.load_model(filepath)
weights = np.array(model.get_weights())
expected_linear_reg = np.zeros(1000)
for m in range(0, 1000):
    #weight = weights[0]
    #bias = weights[1]
    weight = np.random.uniform(low= -1, high = 1, size = (Dim, 16))
    bias = np.random.uniform(low = -1, high= 1, size = 16)

    nr_of_regions = 0
    for j in range(0, width):
        minR = 0
        maxR = 0
        for i in range(0, Dim):
            if weight[i][j] < 0:
                minR = minR + weight[i][j]
            else:
                maxR = maxR + weight[i][j]
        if minR < bias[j] and maxR > bias[j]:
            nr_of_regions = nr_of_regions + 1
    expected_linear_reg[m] = nr_of_regions
#print(expected_linear_reg)
print(np.mean(expected_linear_reg))
