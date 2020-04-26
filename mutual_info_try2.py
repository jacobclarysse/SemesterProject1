import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Constants
min_part = -1
max_part = 1
N = 16 ###Will be different in the end->not a constant anymore!
D = 1
visualisation = 0

### Computation of the differential entropy
H_all = np.zeros(5)
Regions = np.zeros(5)
for i in range(0, 5):
    k0 = 2**i
    M = np.math.factorial(N)/(np.math.factorial(N/k0)**k0)
    P = 1/M*((k0/N)**N)
    H_all[i] = -np.log2(P)
    ### Compute expected nr of linear regions
    Regions[i] = (1-((N/k0-1)/(N/k0))**(N/k0))*N

NH_w = 64*np.ones(5)

Differential_entropy = NH_w - H_all
print(Differential_entropy)
print(Regions)

plt.figure(1)
plt.plot(Regions, Differential_entropy, '*')
plt.xlabel("E[R]")
plt.ylabel("WH(w)-H(w_1,...,w_W")
plt.title("Example 3, d=1, W=16")
plt.show()