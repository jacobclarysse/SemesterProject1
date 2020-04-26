import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

e1 = np.zeros(4)
e1_v = np.zeros(4)
e2 = np.zeros(4)
e2_v = np.zeros(4)
e3 = np.zeros(4)
e3_v = np.zeros(4)

def activations(weights, p):
    if p ==16:
        De = 2
    elif p == 8:
        De = 4
    elif p == 4:
        De = 8
    elif p == 2:
        De = 16
    ranges_a = np.zeros((p,De))
    ranges_c = np.ones((p,De))
    s = 0
    for D in range(0,De):
        if D == 0:
            w = weights[0][0]
            b = weights[1]
            for i in range(0,p):
                if w[i] > b[i] and b[i] > 0:
                    s = s + 1
                elif w[i] < b[i] and b[i] < 0:
                    s = s + 1
                if w[i]>0:
                    ranges_c[i][1] = w[i]
                else:
                    ranges_c[i][1] = 0
                    ranges_a[i][1] = 1
            continue
        w = weights[2*D]
        b = weights[2*D+1]
        for i in range(0, p):
            maxi = 0
            mini = 0
            for j in range(0, p):
                if w[j][i] > 0:
                    maxi = maxi + ranges_c[i][D]*w[i][j]
                    mini = mini + ranges_a[i][D]*w[j][i]
                else:
                    mini = mini + w[j][i]*ranges_c[i][D]
                    maxi = maxi + w[j][i]*ranges_a[i][D]
            if D != De-1:
                ranges_c[i][D+1] = maxi
                ranges_a[i][D+1] = mini
            if b[i] > mini and b[i] < maxi:
                s = s + 1
    return s

results = np.zeros((3,4,50))
for i in range(1,51):
    for a in range(0,4):
        if a == 0:
            p = 16
            D = 2
        elif a == 1:
            p=8
            D = 4
        elif a == 2:
            p=4
            D = 8
        elif a == 3:
            p = 2
            D = 16
        for e in range(1,4):
            name = "Experiment2/D="+str(D)+str(i)+"-model-0"+str(e)+".hdf5"
            model = keras.models.load_model(name)
            weights = model.get_weights()
            results[e-1][a][i-1]= activations(weights, p)
    print(i)

for a in range(0,3):
    e1[a] = np.mean(results[0][a][:])
    e1_v[a] = np.sqrt(np.var(results[0][a][:]))
    e2[a] = np.mean(results[1][a][:])
    e2_v[a] = np.sqrt(np.var(results[1][a][:]))
    e3[a] = np.mean(results[2][a][:])
    e3_v[a] = np.sqrt(np.var(results[2][a][:]))

x = np.array([2,4,8,16])
plt.figure(1)
plt.errorbar(x, e1, yerr=e1_v, fmt='b',linestyle='solid')
plt.errorbar(x, e2, yerr=e2_v, fmt='r', linestyle='dashed')
plt.errorbar(x, e3, yerr=e3_v, fmt='g', linestyle='dashdot')
plt.xlabel("Depth")
plt.ylabel("E[A]")
plt.title("E[A] vs W/D, 32 neurons")
plt.legend({"epoch 1", "epoch 2", "epoch 3"})
plt.show()
