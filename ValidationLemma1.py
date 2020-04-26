import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exp_emp = np.zeros(10)
std_dev = np.zeros(10)
formula_mean = np.zeros(10)
def bin(n, k):
        return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))
c_s = np.zeros(10)
#constants
for iterator in range(0, 10):
    d = 16 #the dimention

    c = (iterator+4)
    c_s[iterator] = c
    a = 0.3*c
    Exp_act = 0
    for i in range(0, d+1):
        Exp_act = Exp_act + bin(d, i)*(1/(np.math.factorial(i+1)*c**i)+1/(np.math.factorial(d-i+1)*a**(d-i)))
    Exp_act = 1 - Exp_act*(1/2)**(d+1)
    formula_mean[iterator] = Exp_act
    print("Expected number of regions formula: "+str(Exp_act))

    N = 100000
    Empirecal_average = 0
    weights = np.random.uniform(-1,1,(N, d))
    biases = np.random.uniform(-1, 1, N)
    Nr_regions = np.zeros(N)
    for i in range(0, N):
        if biases[i] < 0:
            w_min = 0
            for j in range(0, d):
                if (weights[i][j] < 0):
                    w_min = w_min + a*weights[i][j]
            if w_min < biases[i]:
                Empirecal_average = Empirecal_average + 1
                Nr_regions[i] = 1
            else: Nr_regions[i] = 0
        elif biases[i] > 0:
            w_min = 0
            for j in range(0, d):
                if (weights[i][j] > 0):
                        w_min = w_min + c*weights[i][j]
            if w_min > biases[i]:
                Empirecal_average = Empirecal_average + 1
                Nr_regions[i] = 1
            else: Nr_regions[i] = 0

    Empirecal_average = Empirecal_average/N
    exp_emp[iterator] = Empirecal_average
    std_dev[iterator] = np.var(Nr_regions)
    print("Empirical average: "+str(Empirecal_average))

d_m = c_s   
plt.figure(1)
plt.plot(d_m, formula_mean, 'o')
plt.errorbar(d_m, exp_emp, yerr = std_dev, fmt = '*')
plt.xlabel("positive c, a=-0.3*c")
plt.ylabel("E[|A|]")
plt.title("Empirical vs Formula E[|A|]")
plt.legend({"formula", "empirical"})
plt.show()