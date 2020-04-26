import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Define fs1, fs2, fs3, fs4!
fs1 = np.zeros(8)
fs2 = np.zeros(8)
fs3 = np.zeros(8)
fs4 = np.zeros(8)

for i in range(0,2):
    fs1[i] = 1/2
    fs2[i] = 1/2
    fs3[i] = 1/2
    fs4[i] = 1/2

###Compute -4H_w
f_w = 1/4*(fs1+fs2+fs3+fs4)
H_w = 0
S_f_w = np.size(f_w)
for i in range(0, S_f_w):
    if f_w[i] != 0:
        H_w = H_w + f_w[i]*np.log2(f_w[i])
H_w = -4*H_w
print(H_w)

###Compute H_w1_w2_w3_w4

f_w1_w2 = np.zeros((S_f_w, S_f_w))
f_w1_w3 = np.zeros((S_f_w, S_f_w))
f_w1_w4 = np.zeros((S_f_w, S_f_w))
f_w2_w3 = np.zeros((S_f_w, S_f_w))
f_w2_w4 = np.zeros((S_f_w, S_f_w))
f_w3_w4 = np.zeros((S_f_w, S_f_w))

for i in range(0, S_f_w):
    for j in range(0, S_f_w):
        f_w1_w2[i][j] = fs1[i]*fs2[j]
        f_w1_w3[i][j] = fs1[i]*fs3[j]
        f_w1_w4[i][j] = fs1[i]*fs4[j]
        f_w2_w3[i][j] = fs2[i]*fs3[j]
        f_w2_w4[i][j] = fs2[i]*fs4[j]
        f_w3_w4[i][j] = fs3[i]*fs4[j]

f_w1_w2 = f_w1_w2 + np.transpose(f_w1_w2)
f_w1_w3 = f_w1_w3 + np.transpose(f_w1_w3)
f_w1_w4 = f_w1_w4 + np.transpose(f_w1_w4)
f_w2_w3 = f_w2_w3 + np.transpose(f_w2_w3)
f_w2_w4 = f_w2_w4 + np.transpose(f_w2_w4)
f_w3_w4 = f_w3_w4 + np.transpose(f_w3_w4)

f1 = np.zeros((S_f_w, S_f_w, S_f_w, S_f_w))
f2 = np.zeros((S_f_w, S_f_w, S_f_w, S_f_w))
f3 = np.zeros((S_f_w, S_f_w, S_f_w, S_f_w))
f4 = np.zeros((S_f_w, S_f_w, S_f_w, S_f_w))
f5 = np.zeros((S_f_w, S_f_w, S_f_w, S_f_w))
f6 = np.zeros((S_f_w, S_f_w, S_f_w, S_f_w))

for i in range(0, S_f_w):
    for j in range(0, S_f_w):
        for m in range(0, S_f_w):
            for l in range(0, S_f_w):
                f1[i][j][m][l] = f_w1_w2[i][j]*f_w3_w4[m][l]
                f2[i][j][m][l] = f_w1_w3[i][j]*f_w2_w4[m][l]
                f3[i][j][m][l] = f_w1_w4[i][j]*f_w2_w3[m][l]
                f4[i][j][m][l] = f_w2_w3[i][j]*f_w1_w4[m][l]
                f5[i][j][m][l] = f_w2_w4[i][l]*f_w1_w3[m][l]
                f6[i][j][m][l] = f_w3_w4[i][l]*f_w1_w2[m][l]
f_w1_w2_w3_w4 = 1/24*(f1+f2+f3+f4+f5+f6)
#print(f_w1_w2_w3_w4)
H_w1_w2_w3_w4 = 0
for i in range(0, S_f_w):
    for j in range(0, S_f_w):
        for m in range(0, S_f_w):
            for l in range(0, S_f_w):
                if f_w1_w2_w3_w4[i][j][m][l] != 0:
                    H_w1_w2_w3_w4 = H_w1_w2_w3_w4 + f_w1_w2_w3_w4[i][j][m][l]*np.log2(f_w1_w2_w3_w4[i][j][m][l])
H_w1_w2_w3_w4 = -H_w1_w2_w3_w4
print(H_w1_w2_w3_w4)