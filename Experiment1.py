import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mse_ep1 = np.zeros(5)
mse_var1 = np.zeros(5)
mse_ep2 = np.zeros(5)
mse_var2 = np.zeros(5)
mse_ep3 = np.zeros(5)
mse_var3 = np.zeros(5)

actep1 = np.zeros(5)
actepvar1 = np.zeros(5)
actep2 = np.zeros(5)
actepvar2 = np.zeros(5)
actep3 = np.zeros(5)
actepvar3 = np.zeros(5)

activation = np.zeros((5, 300,3))
mses = np.zeros((5, 300,3))

def mse_c(model, D):
    x = np.zeros((D, 10000))
    for i in range(0, D):
        x[i][:] = np.array(np.linspace(0,1, 10000))
    yexact = np.sin(2*np.pi*np.linspace(0,1,10000))
    ypred = model.predict(np.transpose(x))
    mse = np.mean((ypred-yexact)**2)
    return mse

def activations(weights, D):
    if D == 1:
        w = weights[0][0]
        b = weights[1]
        s = 0
        for i in range(0,16):
            if w[i] > b[i] and b[i] > 0:
                s = s + 1
            elif w[i] < b[i] and b[i] < 0:
                s = s + 1
        return s
    w = weights[0]
    print(np.shape(w))
    b = weights[1]
    s = 0
    for i in range(0, 16):
        maxi = 0
        mini = 0
        for j in range(0, D):
            if w[j][i] > 0:
                maxi = maxi + w[j][i]
            else:
                mini = mini + w[j][i]
        if b[i] > mini and b[i] < maxi:
            s = s + 1
    return s

for i in range(1, 301):
    for ep in range(1,4):
        for d in range(0, 5):
            D = 2**(d)
            filepath = "Experiment1/D"+str(D)+"_"+str(i)+"-model-0"+str(ep)+".hdf5"
            model = keras.models.load_model(filepath)
            if D == 1:
                D = 2
            elif D == 2:
                D = 1
            weights = model.get_weights()
            activation[d][i-1][ep-1] = activations(weights, D)
            mses[d][i-1][ep-1] = mse_c(model, D)
    print(i)

for i in range(0, 5):
    mse_ep1[i] = np.mean(mses[i][:][1])
    mse_var1[i] = np.sqrt(np.var(mses[i][:][1]))
    mse_ep2[i] = np.mean(mses[i][:][2])
    mse_var2[i] = np.sqrt(np.var(mses[i][:][2])) 
    mse_ep3[i] = np.mean(mses[i][:][3])
    mse_var3[i] = np.sqrt(np.var(mses[i][:][3]))   

    actep1[i] = np.mean(activation[i][:][1])
    actepvar1[i] = np.sqrt(np.var(activation[i][:][1]))
    actep2[i] = np.mean(activation[i][:][2])
    actepvar2[i] = np.sqrt(np.var(activation[i][:][2]))
    actep3[i] = np.mean(activation[i][:][3])
    actepvar3[i] = np.sqrt(np.var(activation[i][:][3]))

x_ax = np.array([1,2,4,8,16])
plt.figure(1)
plt.errorbar(x_ax, actep1, yerr=actepvar1, fmt="*")
plt.errorbar(x_ax, actep2, yerr=actepvar2, fmt="+")
plt.errorbar(x_ax, actep3, yerr=actepvar3, fmt=".")
plt.xlabel("d")
plt.ylabel("E[|A|]")
plt.title("|A| vs Increasing d")
plt.legend({"epoch 1", "epoch 2", "epoch 3"})

plt.figure(2)
plt.errorbar(x_ax, mse_ep1, yerr=mse_var1, fmt="*")
plt.errorbar(x_ax, mse_ep2, yerr=mse_var2, fmt="+")
plt.errorbar(x_ax, mse_ep3, yerr=mse_var3, fmt=".")
plt.xlabel("d")
plt.ylabel("mse")
plt.title("mse vs Increasing d")
plt.legend({"epoch 1", "epoch 2", "epoch 3"})

plt.show()





