import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bucket_width = 0.2
number_of_buckets = 40 #integer
center = 0
width = 8
nr_models = 100

reg = np.zeros((3, 3))
H_d = np.zeros((3, 3))

for iteration in range(0,3):
    ###Extract weights from models
    for epoch in range(1, 4):
        if iteration == 0:
            name_pre = "SQModels_1D_D1_W16/RELU_1D_D1_V"
        elif iteration == 1: 
            name_pre = "SQModels_1D_D1_W16_2/RELU_1D_D1_V"
        else: name_pre = "SQModels_1D_D1_W16_3/RELU_1D_D1_V"

        name_end = "-model-0"+str(epoch)+".hdf5"

        weights = np.zeros((nr_models, width))
        for i in range(0, nr_models):
            model = keras.models.load_model(name_pre+str(i)+name_end)
            weight = model.get_weights()
            weights[i][:] = weight[0][0]
        print("extracted all the weights")

        f_w1_w16 = np.zeros((number_of_buckets, number_of_buckets, number_of_buckets, number_of_buckets, number_of_buckets, number_of_buckets, number_of_buckets, number_of_buckets)) #fix
        nr_b = number_of_buckets/2
        f_w = np.zeros((width, number_of_buckets))

        for mod in range(0, nr_models):
            b = np.zeros(width)
            for i in range(0, width):
                bucket = int(np.floor((weights[mod][i]-center)/bucket_width+nr_b))
                if bucket < 0:
                    bucket = 0
                    f_w[i][0] = f_w[i][0] + 1
                elif bucket > number_of_buckets-1:
                    bucket = number_of_buckets-1
                    f_w[i][number_of_buckets-1] = f_w[i][number_of_buckets-1]+1
                b[i] = bucket
                f_w[i][bucket] = f_w[i][bucket]+1
            f_w1_w16[int(b[0])][int(b[1])][int(b[2])][int(b[3])][int(b[4])][int(b[5])][int(b[6])][int(b[7])] = f_w1_w16[int(b[0])][int(b[1])][int(b[2])][int(b[3])][int(b[4])][int(b[5])][int(b[6])][int(b[7])]+1

        f_w1_w16 = 1/(nr_models)*f_w1_w16
        f_w = 1/(nr_models)*f_w

        H_w1_W16 = 0
        E_Regions = 0
        H_W = 0

        for a in range(0, number_of_buckets):
            print(str(a))
            for b in range(0, number_of_buckets):
                for c in range(0, number_of_buckets):
                    for d in range(0, number_of_buckets):
                        for e in range(0, number_of_buckets):
                            for f in range(0, number_of_buckets):
                                for g in range(0, number_of_buckets):
                                    for h in range(0, number_of_buckets):
                                        if f_w1_w16[a][b][c][d][e][f][g][h] != 0:
                                            H_w1_W16 = H_w1_W16 + f_w1_w16[a][b][c][d][e][f][g][h]*np.log2(f_w1_w16[a][b][c][d][e][f][g][h])
        H_w1_W16 = -H_w1_W16
        for j in range(0, width):
            for i in range(0, number_of_buckets):
                if f_w[j][i] != 0:
                    H_W = H_W + f_w[j][i]*np.log2(f_w[j][i])
        H_W = -H_W
        for i in range(0, number_of_buckets):
            E_Regions = E_Regions + 1 - (1-f_w[0][i])*(1-f_w[1][i])*(1-f_w[2][i])*(1-f_w[3][i])*(1-f_w[4][i])*(1-f_w[5][i])*(1-f_w[6][i])*(1-f_w[7][i])
        reg[iteration][epoch-1] = E_Regions
        H_d[iteration][epoch-1] = H_W-H_w1_W16

H_d_M = np.zeros(3)
H_d_stdvar = np.zeros(3)

reg_M = np.zeros(3)
reg_stdvar = np.zeros(3)

for i in range(0, 3):
    H_d_M[i] = np.mean(H_d[:][i])
    H_d_stdvar[i] = np.sqrt(np.var(H_d[:][i]))

    reg_M[i] = np.mean(reg[:][i])
    reg_stdvar[i] = np.sqrt(np.var(reg[:][i]))

###Visualization
plt.figure(1)
plt.errorbar(H_d_M, reg_M, yerr= reg_stdvar, fmt = 'o')
plt.xlabel("diff entropy")
plt.ylabel("expected nr regions")
plt.title("width 8 method 6 Square")
plt.show()