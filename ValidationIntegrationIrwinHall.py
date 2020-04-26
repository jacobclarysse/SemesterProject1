import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### We try to validate the integration of the CDF of the Irwin Hall distribution.
def bin(n, k):
        return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))

N = 16
N_th = np.zeros(16)
N_me = np.zeros(16)
N_var = np.zeros(16)
c = 0.15
for n in range(1, N+1):
    E = 0
    round_c = np.floor(1/c)
    if n*c > 1:
        for m in range(0, int(round_c)):
            for k in range(0, m+1):
                E = E + c*(-1)**k*bin(n,k)*((m+1-k)**(n+1)-(m-k)**(n+1))/(np.math.factorial(n+1))
        for k in range(0, int(round_c+1)):
            E = E +c*(-1)**k*bin(n,k)*((1/c-k)**(n+1)-(round_c-k)**(n+1))/(np.math.factorial(n+1))
    else:
        for m in range(0, n):
            for k in range(0, m+1):
                E = E + c*(-1)**k*bin(n,k)*((m+1-k)**(n+1)-(m-k)**(n+1))/(np.math.factorial(n+1))
        E = E + (1-n*c)
    N_th[n-1] = E
T = 10000


for n in range(1, N+1):
    m = np.zeros(10)
    for means in range(0, 10):
        weights = np.random.uniform(0,c,(T, n))
        biases = np.random.uniform(0, 1, T)
        tests = np.zeros(T)
        for i in range(0, T):
            if sum(weights[i][:]) < biases[i]:
                tests[i] = 1
        m[means] = np.mean(tests)        
    N_me[n-1] = np.mean(m)
    N_var[n-1] = np.sqrt(np.var(m))


print(N_th)
print(N_me)

X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
plt.figure(1)
plt.errorbar(X, N_me, yerr=N_var, fmt=".")
plt.plot(X, N_th, "r-")
plt.xlabel("Dimension")
plt.ylabel("P[w<bi]")
plt.title("Validation Integration Irwin Hall distribution: Lemma 2")
plt.legend({"Empirical","Formula"})
plt.show()
