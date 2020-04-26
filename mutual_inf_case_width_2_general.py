import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


diff_entropies = np.zeros(6)
exp_regions = np.zeros(6)
#Element of [0, 5]
for num in range(0, 6):
    starting_point = num #We will be able to iterate over this in order to get different correlations
    ### We have the case widht = 2

    #### Define fs1 and fs2! -> This should be kept flexible!
    fs1 = np.zeros(10)
    fs2 = np.zeros(10)

    for i in range(starting_point, starting_point+5):
        fs1[i] = 1/5
    for j in range(5, 10):
        fs2[j] = 1/5

    ###Compute 2H_w
    f_w = 1/2*(fs1+fs2)

    H_w = 0
    for i in range(0, np.size(f_w)):
        if f_w[i] != 0:
            H_w =H_w+ f_w[i]*np.log2(f_w[i])
    H_w = -2*H_w
    print(H_w)

    buckets_size = np.size(f_w)
    ###Compute H_w1_w2
    f_w1_w2 = np.zeros((buckets_size, buckets_size))
    f_w1_w1 = np.zeros((buckets_size, buckets_size))
    f_w2_w2 = np.zeros((buckets_size, buckets_size))
    for i in range(0, buckets_size):
        for j in range(0, buckets_size):
            f_w1_w2[i][j] = fs1[i]*fs2[j]
            f_w1_w1[i][j] = fs1[i]*fs1[j]
            f_w2_w2[i][j] = fs2[i]*fs2[j]
    f_w1_w2 = f_w1_w2 + np.transpose(f_w1_w2)
    f_w1_w2 = 1/2*f_w1_w2

    H_w1_w2 = 0
    for i in range(0, buckets_size):
        for j in range(0, buckets_size):
            if f_w1_w2[i][j] != 0:
                H_w1_w2 = H_w1_w2 + f_w1_w2[i][j]*np.log2(f_w1_w2[i][j])
    H_w1_w2 = -H_w1_w2
    H_diff = H_w - H_w1_w2

    #print("difference entropy: "+str(H_diff))

    ###Compute the expected number of linear regions
    Regions = 0
    for i in range(0, buckets_size):
        Regions = Regions + 1 - (1-fs1[i])*(1-fs2[i])
    #print(Regions)
    diff_entropies[num] = H_diff
    exp_regions[num] = Regions

print(diff_entropies)
print(exp_regions)

plt.figure(1)
plt.plot(diff_entropies, exp_regions, '*')
plt.xlabel("I(w1;w2)")
plt.ylabel("E[R]")
plt.title("W2 Sliding uniform window correlation")
plt.show()