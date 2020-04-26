import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exp_emp = np.zeros(5)
std_dev = np.zeros(5)
formula_mean = np.zeros(5)
def bin(n, k):
        return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))
c_s = np.zeros(5)
#constants
for iterator in range(0, 5):
    dim = 16 #the dimention

    c = (iterator+1)/8
    c_s[iterator] = c
    a = 0.9*c
    Exp = 1
    round_c = np.floor(1/c)
    round_a = np.floor(1/a)
    for d in range(1, dim+1):
        E = 0
        if d*c > 1:
            for m in range(0, int(round_c)):
                for k in range(0, m+1):
                    E = E + c*(-1)**k*bin(d,k)*((m+1-k)**(d+1)-(m-k)**(d+1))/(np.math.factorial(d+1))
            for k in range(0, int(round_c+1)):
                E = E +c*(-1)**k*bin(d,k)*((1/c-k)**(d+1)-(round_c-k)**(d+1))/(np.math.factorial(d+1))
        else:
            for m in range(0, d):
                for k in range(0, m+1):
                    E = E + c*(-1)**k*bin(d,k)*((m+1-k)**(d+1)-(m-k)**(d+1))/(np.math.factorial(d+1))
            E = E + (1-d*c)
        if a*d > 1:
            for m in range(0, int(round_a)):
                for k in range(0, m+1):
                    E = E + a*(-1)**k*bin(d,k)*((m+1-k)**(d+1)-(m-k)**(d+1))/(np.math.factorial(d+1))
            for k in range(0, int(round_a+1)):
                E = E +a*(-1)**k*bin(d,k)*((1/a-k)**(d+1)-(round_a-k)**(d+1))/(np.math.factorial(d+1))
        else:
            for m in range(0, d):
                for k in range(0, m+1):
                    E = E + a*(-1)**k*bin(d,k)*((m+1-k)**(d+1)-(m-k)**(d+1))/(np.math.factorial(d+1))
            E = E + (1-d*a)
        Exp = Exp + E*bin(dim, d)
    Exp = 1 - Exp*((1/2)**(dim+1))
    formula_mean[iterator] = Exp
    print("Expected number of regions formula: "+str(Exp))

    N = 100000
    m = np.zeros(10)
    for means in range(0, 10):
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
                    Nr_regions[i] = 1
                else: Nr_regions[i] = 0
            elif biases[i] > 0:
                w_min = 0
                for j in range(0, d):
                    if (weights[i][j] > 0):
                            w_min = w_min + c*weights[i][j]
                if w_min > biases[i]:
                    Nr_regions[i] = 1
                else: Nr_regions[i] = 0
        m[means] = np.mean(Nr_regions)
    exp_emp[iterator] = np.mean(m)
    std_dev[iterator] = np.var(m)
    print("Empirical average: "+str(np.mean(m)))

d_m = c_s   
plt.figure(1)
plt.plot(d_m, formula_mean, 'o')
plt.errorbar(d_m, exp_emp, yerr = np.sqrt(std_dev), fmt = '*')
plt.xlabel("positive c, a=-0.9*c")
plt.ylabel("E[|A|]")
plt.title("Empirical vs Formula E[|A|]")
plt.legend({"formula", "empirical"})
plt.show()