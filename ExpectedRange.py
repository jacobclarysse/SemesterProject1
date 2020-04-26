import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = 1
W = 4
N = 10
def bin(n, k):
    return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))

c_cur = 1
for D in range(1, N):
    c_old = 0
    for i in range(1, W+1):
        c_old = c_old + bin(W,i)*c_cur*0.5*i*(1/2)**W
    c_cur = c_old
    print(c_old)