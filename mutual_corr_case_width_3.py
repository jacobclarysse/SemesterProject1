import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###Idea of simultaneously enlarging windows uniformly
exp_reg = np.zeros(8)
diff_entropy = np.zeros(8)
for enlarge in range(1, 9):
    ###Define fs1, fs2, fs3
    fs1 = np.zeros(9)
    fs2 = np.zeros(9)
    fs3 = np.zeros(9)
    delta_enl = int(np.floor(((9-enlarge)/2)))
    for i in range(0, enlarge):
        fs1[i] = 1/enlarge
    for i in range(delta_enl, delta_enl+enlarge):
        fs2[i] = 1/enlarge
    for i in range(0, enlarge):
        fs3[8-i] = 1/enlarge

    ###Compute 3*H_w
    f_w = (fs1+fs2+fs3)/3
    H_w = 0
    S_f_w = np.size(f_w)
    for i in range(0, S_f_w):
        if f_w[i] != 0:
            H_w = H_w + f_w[i]*np.log2(f_w[i])
    H_w = -3*H_w
    ###Compute H_w1_w2_w3
    f_w1_w2_w3 = np.zeros((S_f_w, S_f_w, S_f_w))
    ##We compute the lower dimentional matrixes and then add
    fs1_w2_w3 = np.zeros((S_f_w, S_f_w))
    fs2_w2_w3 = np.zeros((S_f_w, S_f_w))
    fs3_w2_w3 = np.zeros((S_f_w, S_f_w))
    for i in range(0, S_f_w):
        for j in range(0, S_f_w):
            fs1_w2_w3[i][j] = fs2[i]*fs3[j]
            fs2_w2_w3[i][j] = fs1[i]*fs3[j]
            fs3_w2_w3[i][j] = fs1[i]*fs2[j]
    fs1_w2_w3 = fs1_w2_w3+np.transpose(fs1_w2_w3)
    fs2_w2_w3 = fs2_w2_w3+np.transpose(fs2_w2_w3)
    fs3_w2_w3 = fs3_w2_w3+np.transpose(fs3_w2_w3)

    fs1_w1 = np.zeros((S_f_w, S_f_w, S_f_w))
    fs1_w2 = np.zeros((S_f_w, S_f_w, S_f_w))
    fs1_w3 = np.zeros((S_f_w, S_f_w, S_f_w))

    for i in range(0, S_f_w):
        fs1_w1[i][:][:] = fs1[i]*fs1_w2_w3 
        fs1_w2[i][:][:] = fs2[i]*fs2_w2_w3
        fs1_w3[i][:][:] = fs3[i]*fs3_w2_w3
    f_w1_w2_w3 = (fs1_w1 + fs1_w2 + fs1_w3)/6

    H_w1_w2_w3 = 0
    for i in range(0, S_f_w):
        for j in range(0, S_f_w):
            for m in range(0, S_f_w):
                if f_w1_w2_w3[i][j][m] != 0:
                    H_w1_w2_w3 = H_w1_w2_w3 + f_w1_w2_w3[i][j][m]*np.log2(f_w1_w2_w3[i][j][m])
    H_w1_w2_w3 = -H_w1_w2_w3

    ###Diff entropy
    H_diff = H_w-H_w1_w2_w3

    ###Compute expected number of linear regions
    Regions = 0

    for i in range(0, S_f_w):
        Regions = Regions + 1 - (1-fs1[i])*(1-fs2[i])*(1-fs3[i])

    #print(Regions)
    #print(H_diff)
    exp_reg[enlarge-1] = Regions
    diff_entropy[enlarge-1] = H_diff

print(exp_reg)
print(diff_entropy)

plt.figure(1)
plt.plot(diff_entropy, exp_reg, '*')
plt.xlabel("diff entropy")
plt.ylabel("exp number of regions")
plt.title("Correlation W3 simult enlarging uniform Windows")
plt.show()
