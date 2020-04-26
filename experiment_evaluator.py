import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

meanReg = np.zeros(5)
RegVar = np.zeros(5)
meanCost = np.zeros(5)
CostVar = np.zeros(5)

N = 100
reg = np.zeros(100)
costs = np.zeros(100)
Test = np.random.uniform(low= 0, high = 1, size = 10000)
y = np.sin(2*np.pi*Test)
###First D1
for i in range(0, N):
    model = keras.models.load_model("Dim_Experiment/D1/"+str(i)+"-model-03.hdf5")
    ypred = model.predict(Test)
    costs[i] = np.mean((y-ypred)**2)
    weights = model.get_weights()
    weight = weights[0][0]
    bias = weights[1]
    for j in range(0, 16):
        if weight[j] < bias[j] and bias[j] < 0:
            reg[i] =reg[i] + 1
        elif weight[j] > bias[j] and bias[j] > 0:
            reg[i] =reg[i] + 1

print("Done with D = 1")
meanReg[0] = np.mean(reg)
RegVar[0] = np.var(reg)
meanCost[0] = np.mean(costs)
CostVar[0] = np.var(costs)
print(np.mean(reg))
print(np.mean(costs))

reg = np.zeros(100)
costs = np.zeros(100)
Test = np.random.uniform(low= 0, high = 1, size = (10000, 2))
y = np.sin(2*np.pi*Test[0]*Test[1])
d = 2
###First D1
for i in range(0, N):
    model = keras.models.load_model("Dim_Experiment/D2/"+str(i)+"-model-03.hdf5")
    ypred = model.predict(Test)
    costs[i] = np.mean((y-ypred)**2)
    weights = model.get_weights()
    weight = weights[0]
    bias = weights[1]
    for j in range(0, 16):
        w_min = 0
        w_max = 0
        for m in range(0, d):
            if weight[m][j] < 0:
                w_min = w_min + weight[m][j]
            else:
                w_max = w_max + weight[m][j]
        if w_min < bias[j] and w_max > bias[j]:
            reg[i] =reg[i] + 1

print("Done with Dimension 2")
meanReg[1] = np.mean(reg)
RegVar[1] = np.var(reg)
meanCost[1] = np.mean(costs)
CostVar[1] = np.var(costs)
print(np.mean(reg))
print(np.mean(costs))

reg = np.zeros(100)
costs = np.zeros(100)
d = 4
Test = np.random.uniform(low= 0, high = 1, size = (10000, d))
y = np.sin(2*np.pi*Test[0]*Test[1]*Test[2]*Test[3])
###First D1
for i in range(0, N):
    model = keras.models.load_model("Dim_Experiment/D4/"+str(i)+"-model-03.hdf5")
    ypred = model.predict(Test)
    costs[i] = np.mean((y-ypred)**2)
    weights = model.get_weights()
    weight = weights[0]
    bias = weights[1]
    for j in range(0, 16):
        w_min = 0
        w_max = 0
        for m in range(0, d):
            if weight[m][j] < 0:
                w_min = w_min + weight[m][j]
            else:
                w_max = w_max + weight[m][j]
        if w_min < bias[j] and w_max > bias[j]:
            reg[i] =reg[i] + 1

print("Done with Dimension 4")
meanReg[2] = np.mean(reg)
RegVar[2] = np.var(reg)
meanCost[2] = np.mean(costs)
CostVar[2] = np.var(costs)
print(np.mean(reg))
print(np.mean(costs))

reg = np.zeros(100)
costs = np.zeros(100)          
d = 8
Test = np.random.uniform(low= 0, high = 1, size = (10000, 8))
y = np.sin(2*np.pi*Test[0]*Test[1]*Test[2]*Test[3]*Test[4]*Test[5]*Test[6]*Test[7])
###First D1
for i in range(0, N):
    model = keras.models.load_model("Dim_Experiment/D8/"+str(i)+"-model-03.hdf5")
    ypred = model.predict(Test)
    costs[i] = np.mean((y-ypred)**2)
    weights = model.get_weights()
    weight = weights[0]
    bias = weights[1]
    for j in range(0, 16):
        w_min = 0
        w_max = 0
        for m in range(0, d):
            if weight[m][j] < 0:
                w_min = w_min + weight[m][j]
            else:
                w_max = w_max + weight[m][j]
        if w_min < bias[j] and w_max > bias[j]:
            reg[i] =reg[i] + 1

print("Done with Dimension 8")
meanReg[3] = np.mean(reg)
RegVar[3] = np.var(reg)
meanCost[3] = np.mean(costs)
CostVar[3] = np.var(costs)
print(np.mean(reg))
print(np.mean(costs))

reg = np.zeros(100)
costs = np.zeros(100)
d = 16
Test = np.random.uniform(low= 0, high = 1, size = (10000, d))
y = np.sin(2*np.pi*Test[0]*Test[1]*Test[2]*Test[3]*Test[4]*Test[5]*Test[6]*Test[7]*Test[8]*Test[9]*Test[10]*Test[11]*Test[12]*Test[13]*Test[14]*Test[15])
###First D1
for i in range(0, N):
    model = keras.models.load_model("Dim_Experiment/D16/"+str(i)+"-model-03.hdf5")
    ypred = model.predict(Test)
    costs[i] = np.mean((y-ypred)**2)
    weights = model.get_weights()
    weight = weights[0]
    bias = weights[1]
    for j in range(0, 16):
        w_min = 0
        w_max = 0
        for m in range(0, d):
            if weight[m][j] < 0:
                w_min = w_min + weight[m][j]
            else:
                w_max = w_max + weight[m][j]
        if w_min < bias[j] and w_max > bias[j]:
            reg[i] =reg[i] + 1

print("Done with Dimension 16")
meanReg[4] = np.mean(reg)
RegVar[4] = np.var(reg)
meanCost[4] = np.mean(costs)
CostVar[4] = np.var(costs)
print(np.mean(reg))
print(np.mean(costs))

###Plot the results
DIM = np.array([1,2,4,8,16])
plt.figure(1)
plt.errorbar(DIM , meanReg, yerr = np.sqrt(RegVar), fmt = '*')
plt.xlabel("Dimensions")
plt.ylabel("Average number of Regions")
plt.title("Average Number of Regions with increasing Dim, W = 16, depth = 1")

plt.figure(2)
plt.errorbar(DIM, meanCost, yerr = np.sqrt(CostVar), fmt = '*')
plt.xlabel("Dimensions")
plt.ylabel("mse")
plt.title("MSE cost vs Dimension increase, W =16, depth = 1")
plt.show()
