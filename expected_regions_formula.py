import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exp_emp = np.zeros(5)
std_dev = np.zeros(5)
formula_mean = np.zeros(16)
def bin(n, k):
        return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))
for dim in range(1,17):
    Exp_act = 0
    d = dim
    for i in range(0, d+1):
        Exp_act = Exp_act + bin(d, i)*1/np.math.factorial(i+1)
    Exp_act = 1 - Exp_act*(1/2)**d
    formula_mean[dim-1] = Exp_act
#constants
for dim in range(0, 5):
    d = 2**dim #the dimention
    N = 100000
    m = np.zeros(100)
    for means in range(0, 100):
        Empirecal_average = 0
        weights = np.random.uniform(-1,1,(N, d))
        biases = np.random.uniform(-1, 1, N)
        Nr_regions = np.zeros(N)
        for i in range(0, N):
            if biases[i] < 0:
                w_min = 0
                for j in range(0, d):
                    if (weights[i][j] < 0):
                        w_min = w_min + weights[i][j]
                if w_min < biases[i]:
                    Empirecal_average = Empirecal_average + 1
                    Nr_regions[i] = 1
                else: Nr_regions[i] = 0
            elif biases[i] > 0:
                w_min = 0
                for j in range(0, d):
                    if (weights[i][j] > 0):
                            w_min = w_min + weights[i][j]
                if w_min > biases[i]:
                    Empirecal_average = Empirecal_average + 1
                    Nr_regions[i] = 1
                else: Nr_regions[i] = 0
        m[means] = np.mean(Nr_regions)
    

    exp_emp[dim] = np.mean(m)
    std_dev[dim] = np.var(m)
    print("Empirical average: "+str(np.mean(m)))

x_form = np.array(range(1,17))
d_m = np.array([1, 2, 4, 8, 16])    
plt.figure(1)
plt.plot(x_form, formula_mean, 'o')
plt.errorbar(d_m, exp_emp, yerr = np.sqrt(std_dev), fmt = '*')
plt.xlabel("Dimension d")
plt.ylabel("E[|A|]")
plt.title("Empirical vs Formula E[|A|]")
plt.legend({"formula", "empirical"})
plt.show()
