import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bin(n, k):
    return np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))

def phi(N, c):
    p = 0
    if N*c > 1:
        round_c = np.floor(1/c)
        for m in range(0, int(round_c)):
            for k in range(0, m+1):
                p = p + bin(N,k)*(-1)**k*((m+1-k)**(N+1)-(m-k)**(N+1))
        for k in range(0, int(round_c+1)):
            p = p + bin(N,k)*(-1)**k*((1/c-k)**(N+1)-(round_c-k)**(N+1))
        p = p*c/np.math.factorial(N+1)
    else:
        for m in range(0, N):
            for k in range(0,m+1):
                p = p + bin(N,k)*c*((-1)**k)*((m+1-k)**(N+1)-(m-k)**(N+1))/np.math.factorial(N+1)
        p = p + 1 - N*c
    return p

def Lemma3(W, a, c):
    P = 1
    for l in range(1, W+1):
        P = P + bin(W,l)*(phi(l,c)+phi(l, a))
    P = P/(2**(W+1))
    return 1 - P

W = 5
alpha = 0.3 
a_v = np.linspace(0.001, 0.01, 1000)
c_v = np.linspace(0.01, 0.02, 1000)
minimum = 100000000000000
for i in range(0, 1000):
    for j in range(0, 1000):
        L = Lemma3(W, a_v[j], c_v[i])
        #print(L)
        c_new = 0
        a_new = 0
        for m in range(0, W+1):
            #L = 1 - phi(m,c_v[i])+phi(W-m, a_v[j])
            c_new = c_new + bin(W,m)*((1/2)**(W+1))*(m*c_v[i]+(W-m)*a_v[j])*(L+(1-L)*alpha)
            a_new = a_new + bin(W,m)*((1/2)**(W+1))*(m*c_v[i]+(W-m)*a_v[j])*(L*alpha+(1-L))
        if np.absolute(c_v[i]-c_new)/(c_v[i]+c_new)+np.absolute(a_v[j]-a_new)/(a_v[j]+a_new) < minimum:
            minimum = np.absolute(c_v[i]-c_new)/(c_v[i]+c_new)+np.absolute(a_v[j]-a_new)/(a_v[j]+a_new)
            print("c "+str(c_new)+" "+str(c_v[i]))
            print("a "+str(a_new)+" "+str(a_v[j]))
            print(np.absolute(c_v[i]-c_new)/(c_v[i]+c_new)+np.absolute(a_v[j]-a_new)/(a_v[j]+a_new))


