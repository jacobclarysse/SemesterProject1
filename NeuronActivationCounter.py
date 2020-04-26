import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#program that counts the number neurons that activate within the input space [a,b]^d with a<=0 and b>0.
#We use Leaky RELU with alpha = 0.3
Smurf = 2
means_1 = np.zeros(Smurf-1)
variances_1 = np.zeros(Smurf-1)
means_2 = np.zeros(Smurf-1)
variances_2 = np.zeros(Smurf-1)
means_3 = np.zeros(Smurf-1)
variances_3 = np.zeros(Smurf-1)
means_4 = np.zeros(Smurf-1)
variances_4 = np.zeros(Smurf-1)
for plop in range(0, 4):
    for DIM in range(1, Smurf):
        results = np.zeros(1000)
        for N in range(0, 1000):
            ###constants
            a = 0
            b = 1
            D = DIM
            d = 2
            if plop == 0:
                width = 3
            elif plop == 1:
                width = 4
            elif plop == 2:
                width = 5
            else: width = 6
            alpha = 0.3

            ###We start with the case we want to research, where we can change dimention and depth
            ## weights and biases
            weights_layer_1 = np.random.uniform(-1, 1, (d, width))
            weights = np.random.uniform(-1, 1 ,(width, width, D-1)) #D>1
            biases = np.random.uniform(-1, 1, (width, D))
            nr_of_activations = 0

            ranges = np.zeros((width, 2)) #the ranges of outputs of the net where we're looking at

            ###Layer 1
            for i in range(0, width):
                w_min = 0
                w_max = 0
                for j in range(0, d):
                    if weights_layer_1[j][i] < 0:
                        w_min = w_min+b*weights_layer_1[j][i]
                        w_max = w_max+a*weights_layer_1[j][i]
                    else:
                        w_min = w_min+a*weights_layer_1[j][i]
                        w_max = w_max+b*weights_layer_1[j][i]
                if w_min < biases[i][0] and w_max > biases[i][0]:
                    nr_of_activations = nr_of_activations + 1
                if w_min < biases[i][0]:
                    ranges[i][0] = alpha*w_min
                else: ranges[i][0] = w_min
                if w_max < biases[i][0]:
                    ranges[i][1] = alpha*w_max
                else: ranges[i][1] = alpha*w_max
            ###Layer 2-D
            for de in range(1, D):
                for i in range(0, width):
                    w_min = 0
                    w_max = 0
                    for j in range(0, width):
                        if weights[i][j][de-1] < 0:
                            w_min = w_min + ranges[i][1]*weights[i][j][de-1]
                            w_max = w_max + ranges[i][0]*weights[i][j][de-1]
                        else:
                            w_min = w_min + ranges[i][0]*weights[i][j][de-1]
                            w_max = w_max + ranges[i][1]*weights[i][j][de-1]
                    if w_min < biases[i][de] and w_max > biases[i][de]:
                        nr_of_activations = nr_of_activations + 1     
                    if w_min < biases[i][de]:
                        ranges[i][0] = alpha*w_min
                    else: ranges[i][0] = w_min
                    if w_max < biases[i][de]:
                        ranges[i][1] = alpha*w_max
                    else: ranges[i][1] = alpha*w_max

            ###Now we compute the probability of a neuron activating by deviding by all the neurons
            Prob = nr_of_activations/(D*width)
            #print(Prob) 
            results[N] = Prob
        if plop == 0:
            means_1[DIM-1] = np.mean(results)
            variances_1[DIM-1] = np.var(results)
        elif plop == 1:
            means_2[DIM-1] = np.mean(results)
            variances_2[DIM-1] = np.var(results)
        elif plop == 2:
            means_3[DIM-1] = np.mean(results)
            variances_3[DIM-1] = np.var(results)
        elif plop == 3:
            means_4[DIM-1] = np.mean(results)
            variances_4[DIM-1] = np.var(results)
    print(plop)


plt.figure(1)
x = np.array(range(1,Smurf))
plt.errorbar(x, means_1, yerr = np.sqrt(variances_1), fmt ="*")
plt.errorbar(x, means_2, yerr = np.sqrt(variances_2), fmt ="^")
plt.errorbar(x, means_3, yerr = np.sqrt(variances_2), fmt ="+")
plt.errorbar(x, means_4, yerr = np.sqrt(variances_2), fmt =".")
plt.xlabel("Depth")
plt.ylabel("P[z_i activates]")
plt.title("W3, W4, W5 and W6, dim 2")
plt.legend({"W3","W4","W5","W6"})
plt.show()           


