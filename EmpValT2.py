import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Cor_1 = np.zeros(6)
Lemma1 = np.zeros(6)

W = 4

def bin(n, k):
        return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))

def phi(W, c):
    s = 0
    round_c = np.floor(1/c)
    if W*c > 1:
        for m in range(0, int(round_c)):
            for k in range(0, m+1):
                s = s+bin(W,k)*((-1)**k)*((m+1-k)**(W+1)-(m-k)**(W+1))
        for k in range(0, int(round_c+1)):
            s = s+ bin(W,k)*(-1)**k*((1/c-k)**(W+1)-(round_c-k)**(W+1))
        s = s*c/np.math.factorial(W+1)
        return s
    else:
        for m in range(0, W):
            for k in range(0, m+1):
                s = s+bin(W,k)*(-1)**k*((m+1-k)**(W+1)-(m-k)**(W+1))
        s = s*c/np.math.factorial(W+1) + 1 - W*c
    return s

def xi(W, a, c, m):
    suml = 1
    a_s = a*np.ones(W)
    for i in range(0, m):
        a_s[i] = c
    for plop in range(1, W+1):
        s = 0
        for j1 in range(1, W+1):
            if j1 == W:
                l = 1 - a_s[j1-1]
                if l < 0:
                    continue
                else:
                    s = s + l**(W+1)
            if plop == 1:
                l = 1 - a_s[j1-1]
                if l < 0:
                    continue
                else:
                    s = s + l**(W+1)
            for j2 in range(j1+1, W+1):
                if plop == 1:
                    break
                if plop == 2:
                    l = 1 - a_s[j1-1]-a_s[j2-1]
                    if l < 0:
                        continue
                    else:
                        s = s + l**(W+1) 
                if j2 == W:
                    l = 1 - a_s[j1-1]-a_s[j2-1]
                    if l < 0:
                        continue
                    else:
                        s = s + l**(W+1)  
                for j3 in range(j2+1, W+1):
                    if plop == 2:
                        break
                    if plop == 3:
                        l = 1 - a_s[j1-1]-a_s[j2-1]-a_s[j3-1]
                        if l < 0:
                            continue
                        else:
                            s = s + l**(W+1)
                    if j3 == W:
                        l = 1 - a_s[j1-1]-a_s[j2-1]-a_s[j3-1]
                        if l < 0:
                            continue
                        else:
                            s = s + l**(W+1)
                    for j4 in range(j3+1, W+1):
                        if plop == 3:
                            break
                        l = 1 - a_s[j1-1]-a_s[j2-1]-a_s[j3-1]-a_s[j4-1]
                        if l < 0:
                            continue
                        else:
                            s = s + l**(W+1)
        suml = suml+s*((-1)**plop)
    suml=suml/(np.math.factorial(W+1)*(c**m)*(a**(W-m)))
    return suml

a = 0.5
ca = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

for i in range(0, 6):
    for m in range(0, W+1):
        Cor_1[i] = Cor_1[i]+(bin(W,m))*(xi(W, a, ca[i], m)+xi(W, a, ca[i], m))
        Lemma1[i] = Lemma1[i]+bin(W,m)*phi(W, (m*ca[i]+(W-m)*a)/W)
    Cor_1[i] = 1-Cor_1[i]/(2**(W+1))
    Lemma1[i] = 1 - Lemma1[i]/(2**W)    

print(Cor_1)
print(Lemma1)

N = 100000
var = np.zeros(6)
mean = np.zeros(6)
bias = np.random.uniform(-1, 1, N)
for i in range(0, 6):
    meansp = np.zeros(10)
    for means in range(0, 10):
        weight = np.random.uniform(-1, 1, (N, W))
        numbers = np.zeros(N)
        for it in range(0, N):
            mins = 0
            maxs = 0
            for wit in range(0, W):
                if weight[it][wit] > 0:
                    mins = mins - weight[it][wit]*a
                    maxs = maxs + weight[it][wit]*ca[i]
                else:
                    mins = mins + weight[it][wit]*ca[i]
                    maxs = maxs - weight[it][wit]*a
            if mins < bias[it] and maxs > bias[it]:
                numbers[it] = 1
        meansp[means] = np.mean(numbers)
    mean[i] = np.mean(meansp)
    var[i] = np.var(meansp)
std_var = np.sqrt(var)
#print(var)
print(mean)
print(std_var)
###Plot all
x = np.array(range(0, 6))
plt.figure(1)
plt.plot(x, Cor_1, '*')
plt.plot(x, Lemma1, '+')
plt.errorbar(x, mean, yerr=std_var, fmt='.')
plt.xlabel("a = 0.5, c= {0.5,0.6,..,1.0}")
plt.ylabel("P[z_i,j activates]")
plt.title("Empirical P[z_i,j activates], exact formula and approximation, W = 4")
plt.legend({"Exact", "Approximation", "Empirical"})
plt.show()