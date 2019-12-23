# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:50:02 2019

@author: Florin Albisoru 

Description: mergem usor si rezolvam iarasi problema din 2005_1_art_19
si mergem foarte usor si babeste pe ea
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# crearea nodurilor 
NE = 20
L = 25
N = 20

x_edges = np.zeros(N+1)
y_edges = np.zeros(N+1)

x_edges[0:int(N/4)] = np.linspace(0,0.03,num=int(N/4),endpoint = False)
x_edges[int(N/4):int(N/2)] = 0.03*np.ones(int(N/4))
x_edges[int(N/2):int(3*N/4)] = np.linspace(0.03,0,num=int(N/4),endpoint=False)

y_edges[int(N/4):int(N/2)] = np.linspace(0,0.03,num=int(N/4),endpoint = False)
y_edges[int(N/2):int(3*N/4)] = 0.03*np.ones(int(N/4))
y_edges[int(3*N/4):N] = np.linspace(0.03,0,num=int(N/4),endpoint=False)

x_nodes = np.zeros(N+L)
y_nodes = np.zeros(N+L)

for i in range(N):
    x_nodes[i] = (x_edges[i]+x_edges[i+1])/2
    y_nodes[i] = (y_edges[i]+y_edges[i+1])/2

x_nodes[N:N+int(L/5)] = x_nodes[0:int(N/4)]
y_nodes[N:N+int(L/5)] = y_nodes[5]*np.ones(int(N/4))
x_nodes[N+int(L/5):N+int(2*L/5)] = x_nodes[0:int(N/4)]
y_nodes[N+int(L/5):N+int(2*L/5)] = y_nodes[6]*np.ones(int(N/4))
x_nodes[N+int(2*L/5):N+int(3*L/5)] = x_nodes[0:int(N/4)]
y_nodes[N+int(2*L/5):N+int(3*L/5)] = y_nodes[7]*np.ones(int(N/4))
x_nodes[N+int(3*L/5):N+int(4*L/5)] = x_nodes[0:int(N/4)]
y_nodes[N+int(3*L/5):N+int(4*L/5)] = y_nodes[8]*np.ones(int(N/4))
x_nodes[N+int(4*L/5):N+L] = x_nodes[0:int(N/4)]
y_nodes[N+int(4*L/5):N+L] = y_nodes[9]*np.ones(int(N/4))

#code to check whether the points are correct written
plt.plot(x_edges, y_edges, 'ro', x_nodes[0:27], y_nodes[0:27], 'bo')
plt.axis([0, 0.05, 0, 0.05])
plt.show()

#punem coordonatele normalelor la domeniu
xnormal = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,-1,-1,-1,-1,-1])
ynormal = np.array([-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])

#punerea conditiilor pe frontiera
index = np.zeros(N)
index[int(3*N/4):N] = np.ones(int(N/4))
u = np.zeros(N)
q = np.zeros(N)
u[0:int(3*N/4)] = 100
q[int(3*N/4):N] = -10000

#find alpha coefficient
Q = np.zeros(N+L)
f = np.zeros((N+L,N+L))
for s in range(N+L):
    Q[s] = 10**7*(x_nodes[s]**2 + y_nodes[s]**2)
    for k in range(N+L):
        r = math.sqrt((x_nodes[k]-x_nodes[s])**2+(y_nodes[k]-y_nodes[s])**2)
        f[s,k] = -(1 + r)
alpha = np.linalg.solve(f,Q)

#find G and H
G = np.zeros((N,N))
H = np.zeros((N,N))

#nodes and weights for the 4 point Gauss quadrature
csi = np.array([0.86113631,-0.86113631,0.33998104,-0.33998104])
wg = np.array([0.34785485,0.34785485,0.65214515,0.65214515])

for i in range(N):
    for j in range(N):
        lj = math.sqrt((x_edges[j+1] - x_edges[j])**2 + (y_edges[j+1] - y_edges[j])**2)
        if i!=j:
            sumaG = 0
            sumaH = 0
            for k in range(4):
                xcsi = (x_edges[j+1] + x_edges[j])/2 + csi[k]*(x_edges[j+1] - x_edges[j])/2
                ycsi = (y_edges[j+1] + y_edges[j])/2 + csi[k]*(y_edges[j+1] - y_edges[j])/2
                r = math.sqrt((xcsi-x_nodes[i])**2 + (ycsi-y_nodes[i])**2)
                rdn = (xcsi-x_nodes[i])*xnormal[j] + (ycsi-y_nodes[i])*ynormal[j]
                sumaH = sumaH + wg[k]*rdn/(r*r)
                sumaG = sumaG + wg[k]*math.log(1/r)
            G[i,j] = sumaG*lj/(4*math.pi)
            H[i,j] = sumaH*lj/(4*math.pi)
        else:
            H[i,j] = -1/2
            G[i,j] = lj/(2*math.pi)*(1 - math.log(lj/2))

# find W and U
U = np.zeros((N,N+L))

for j in range(N):
    for k in range(N+L):            
        r = math.sqrt((x_nodes[k]-x_nodes[j])**2 + (y_nodes[k]-y_nodes[j])**2)
        U[j,k] = (r**2)/4 + (r**3)/9
        
W = np.zeros((N,N+L))

for j in range(N):
    for k in range(N+L):
        r = math.sqrt((x_nodes[k]-x_nodes[j])**2 + (y_nodes[k]-y_nodes[j])**2)
        d = (x_nodes[k]-x_nodes[j])*xnormal[j] + (y_nodes[k]-y_nodes[j])*ynormal[j]
        W[j,k] = -d*(0.5+r/3)

freeterm = np.matmul(G,W) - np.matmul(H,U)
freeterm = np.matmul(freeterm,alpha)

system_matrix = np.zeros((N,N))
aux_matrix = np.zeros((N,N))
known_cond = np.zeros(N)
known_cond[0:15] = u[0:15] 
known_cond[15:20] = q[15:20]
system_matrix[:,0:15] = G[:,0:15]
system_matrix[:,15:20] = -H[:,15:20]
aux_matrix[:,0:15] = H[:,0:15]
aux_matrix[:,15:20] = -G[:,15:20]
freeterm = freeterm + np.matmul(aux_matrix,known_cond)

rez = np.linalg.solve(system_matrix,freeterm)
u[15:20] = rez[15:20]
q[0:15] = rez[0:15]

#calculam doar pentru un singur nod
h_nod = np.zeros(N)
g_nod = np.zeros(N)
u_nod = np.zeros(N+L)

xi = x_nodes[N]
yi = y_nodes[N]
for j in range(N):
    lj = math.sqrt((x_edges[j+1] - x_edges[j])**2 + (y_edges[j+1] - y_edges[j])**2)
    sumaG = 0
    sumaH = 0
    for k in range(4):
        xcsi = (x_edges[j+1] + x_edges[j])/2 + csi[k]*(x_edges[j+1] - x_edges[j])/2
        ycsi = (y_edges[j+1] + y_edges[j])/2 + csi[k]*(y_edges[j+1] - y_edges[j])/2
        r = math.sqrt((xcsi-xi)**2 + (ycsi-yi)**2)
        rdn = (xcsi-xi)*xnormal[j] + (ycsi-yi)*ynormal[j]
        sumaH = sumaH + rdn/(r*r)*wg[k]
        sumaG = sumaG + math.log(1/r)*wg[k]
    g_nod[j] = sumaG*lj/(4*math.pi)
    h_nod[j] = sumaH*lj/(4*math.pi)

for k in range(N+L):
    r = math.sqrt((x_nodes[k] - xi)**2 + (y_nodes[k] - yi)**2)
    u_nod[k] = (r**2)/4 + (r**3)/9  
    
determinat = 0
for j in range(N):
    determinat = determinat + h_nod[j]*u[j] - g_nod[j]*q[j]
    
for k in range(N+L):
    determinat = determinat + alpha[k]*u_nod[k]
    sumpar1 = 0
    sumpar2 = 0
    for j in range(N):
        sumpar1 = sumpar1 + g_nod[j]*W[j,k]
        sumpar2 = sumpar2 + h_nod[j]*U[j,k]
    determinat = determinat + alpha[k]*sumpar1 - alpha[k]*sumpar2
"""            
G_trial = np.zeros((N,N))

from scipy import integrate

for i in range(N):
    xi = x_nodes[i]
    yi = y_nodes[i]
    for j in range(N):
        lj = math.sqrt((x_edges[j+1] - x_edges[j])**2 + (y_edges[j+1] - y_edges[j])**2)
        if i != j :
            cant1 = (x_edges[j+1] + x_edges[j])/2
            cant11 = (x_edges[j+1] - x_edges[j])/2
            cant2 = (y_edges[j+1] + y_edges[j])/2
            cant22 = (y_edges[j+1] - y_edges[j])/2
            f = lambda x : -np.log(np.sqrt((cant1+x*cant11 - xi)**2+(cant2+x*cant22 - yi)**2))
            rez = integrate.fixed_quad(f, -1.0, 1.0, n=4)[0]
            G_trial[i,j] = (l/(4*math.pi))*rez
"""

