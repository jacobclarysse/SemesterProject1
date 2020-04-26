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

N = 20
W = 3
P = 10000
alpha = 0.3
a_v = np.zeros(N)
a_v_var = np.zeros(N)
c_v = np.ones(N)
c_v_var = np.zeros(N)
a_vc = np.zeros(N)
c_vc = np.ones(N)

weights = np.random.uniform(-1,1,(W,W,N,P))
ranges_c = np.ones((W, N, P))
ranges_a = np.zeros((W, N, P))

biasses = np.random.uniform(-1,1, (N,W,P))
for D in range(0, N-1):
    ###Compute with formula
    #L = Lemma3(W, a_vc[D], c_vc[D])
    c_vc[D+1] = 0
    for m in range(0, W+1):
        L2 = bin(W,m)*((1/2)**(W+1))*(phi(W-m, np.absolute(a_vc[D]))+phi(m, c_vc[D]))
        #L = 1 - phi(m,c_v[i])+phi(W-m, a_v[j])
        c_vc[D+1] = c_vc[D+1] + bin(W,m)*((1/2)**(W+1))*(m*c_vc[D]-(W-m)*a_vc[D])*(L2+(1-L2)*alpha)
        a_vc[D+1] = a_vc[D+1] + bin(W,m)*((1/2)**(W+1))*(-(W-m)*c_vc[D]+(m)*a_vc[D])*((L2)*alpha+1-L2)
    ###empirically
    for i in range(0, P):
        for ne in range(0, W):
            w_min = 0
            w_max = 0
            for we in range(0, W):
                if weights[ne][we][D][i] < 0:
                    w_min = w_min + ranges_c[ne][D][i]*weights[ne][we][D][i]
                    w_max = w_max + ranges_a[ne][D][i]*weights[ne][we][D][i]
                else:
                    w_min = w_min + ranges_a[ne][D][i]*weights[ne][we][D][i]
                    w_max = w_max + ranges_c[ne][D][i]*weights[ne][we][D][i]
            if w_min > biasses[D][ne][i]:
                ranges_c[ne][D+1][i] = w_max + biasses[D][ne][i]
                ranges_a[ne][D+1][i] = w_min + biasses[D][ne][i]
            if w_min < biasses[D][ne][i] and w_max > biasses[D][ne][i]:
                ranges_c[ne][D+1][i] = w_max + biasses[D][ne][i]
                ranges_a[ne][D+1][i] = alpha*(w_min + biasses[D][ne][i])
            else:
                ranges_c[ne][D+1][i] = alpha*(w_max + biasses[D][ne][i])
                ranges_a[ne][D+1][i] = alpha*(w_min + biasses[D][ne][i])
    a_v[D+1] = np.mean(ranges_a[0][D+1][:])
    a_v_var[D+1] = np.var(ranges_a[0][D+1][:])
    c_v[D+1] = np.mean(ranges_c[0][D+1][:])
    c_v_var[D+1] = np.var(ranges_c[0][D+1][:])

            
###Plot results
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.figure(1)
plt.plot(x, c_vc, '*')
plt.errorbar(x, c_v, yerr=np.sqrt(c_v_var), fmt = "+")
plt.xlabel("Depth")
plt.ylabel("positive output range")
plt.title("positive output range vs Depth W5")
plt.figure(2)
plt.plot(x, a_vc, '*')
plt.errorbar(x, a_v, yerr=np.sqrt(a_v_var), fmt = "+")
plt.xlabel("Depth")
plt.ylabel("Negative output range")
plt.title("Negative output range vs Depth W5")
plt.show()

    
