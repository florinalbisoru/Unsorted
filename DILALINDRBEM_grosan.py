# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:55:23 2019

@author: Florin Albisoru 

Description : problema lui Grosan
functioneaza
"""

import numpy as np
import math
import matplotlib.pyplot as plt

N = 180
L = 20
NE = N

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)] = np.linspace(0,1,num=int(N/4),endpoint=False)
xedge[int(N/4):int(N/2)] = np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(1,0,num = int(N/4), endpoint=False)

yedge[int(N/4):int(N/2)] = np.linspace(0,1,num=int(N/4),endpoint=False)
yedge[int(N/2):int(3*N/4)] = np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(1,0,num = int(N/4), endpoint=False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)

xnode[0:N] = xedge[0:N]
ynode[0:N] = yedge[0:N]

xnode[N:N+L] = np.linspace(0.01,0.99,num = L)
ynode[N:N+L] = 0.5*np.ones(L)

#conditiile pe frontiera
q = np.zeros(N)
# se da ca u e 0 pe frontiera
u = np.zeros(N)
index = np.ones(N)

#termenul liber
b = (-1)*np.ones(N+L)
F = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        F[k,j] = 1 + r
        
alpha = np.linalg.solve(F,b)

csi = np.array([0.339981,-0.339981,0.861136,-0.861136])
wg = np.array([0.652145,0.652145,0.347855,0.347855])

H = np.zeros((N+L,N))
G = np.zeros((N+L,2*N))

system_matrix = np.zeros((N,N))
system_freeterm = np.zeros(N)

for j1 in range(N+L):
    cc = 0
    for j2 in range(NE):
        if (j2 == NE - 1): k = 0
        else: k = j2 + 1
        lj = math.sqrt((xedge[k] - xedge[j2])**2 + (yedge[k]-yedge[j2])**2)
        h1 = 0
        h2 = 0 
        g1 = 0
        g2 = 0
        if (j1!=j2 and j1!=k):
            for j3 in range(4):
                x = xedge[j2] + (1 + csi[j3])*(xedge[k]-xedge[j2])/2
                y = yedge[j2] + (1 + csi[j3])*(yedge[k]-yedge[j2])/2
                r = math.sqrt((xnode[j1]-x)**2+(ynode[j1]-y)**2)
                pp = -((xnode[j1]-x)*(yedge[j2]-yedge[k]) + (ynode[j1]-y)*(xedge[k]-xedge[j2]))/(r*r*4*math.pi)*wg[j3]
                h1 = h1 + (1-csi[j3])*pp/2
                h2 = h2 + (1+csi[j3])*pp/2
                pp = math.log(1/r)/(4*math.pi)*lj*wg[j3]
                g1 = g1 + (1-csi[j3])*pp/2
                g2 = g2 + (1+csi[j3])*pp/2
            cc = cc - h1 - h2
        else:
            ge = lj*(3/2 - math.log(lj))/(4*math.pi)
        if (j2 == j1): g1 = ge
        if (k == j1): g2 = ge
        H[j1,j2] = H[j1,j2] + h1
        H[j1,k] = H[j1,k] + h2
        G[j1,2*j2] = g1
        G[j1,2*j2+1] = g2
        if (j1 < N):
            if index[j2] == 2:
                system_freeterm[j1] = system_freeterm[j1] + q[j2]*g1
                system_matrix[j1,j2] = system_matrix[j1,j2] + h1
            else:
                system_freeterm[j1] = system_freeterm[j1] - u[j2]*h1
                system_matrix[j1,j2] = system_matrix[j1,j2] - g1
            if index[k] == 2:
                system_freeterm[j1] = system_freeterm[j1] + q[k]*g2
                system_matrix[j1,k] = system_matrix[j1,k] + h2
            else:
                system_freeterm[j1] = system_freeterm[j1] - u[k]*h2
                system_matrix[j1,k] = system_matrix[j1,k] - g2
    if (j1 < N): 
        H[j1,j1] = cc
        if (index[j1] == 2):
            system_matrix[j1,j1] = cc
        else:
            system_freeterm[j1] = system_freeterm[j1] - u[j1]*cc
            
d = np.zeros(N+L)
qh = np.zeros(2*NE)
uh = np.zeros(N)

for k in range(NE):
    if (k == NE - 1): i = 0
    else: i = k + 1
    lek = math.sqrt((xedge[i] - xedge[k])**2 + (yedge[i]-yedge[k])**2) 
    for j in range(N+L):
        r1 = math.sqrt((xedge[k] - xnode[j])**2 + (yedge[k]-ynode[j])**2)
        r2 = math.sqrt((xedge[i] - xnode[j])**2 + (yedge[i]-ynode[j])**2)
        qh1 = -(0.5+r1/3)*((xedge[k]-xnode[j])*(yedge[k]-yedge[i])+(yedge[k]-ynode[j])*(xedge[i]-xedge[k]))/lek
        qh2 = -(0.5+r2/3)*((xedge[i]-xnode[j])*(yedge[k]-yedge[i])+(yedge[i]-ynode[j])*(xedge[i]-xedge[k]))/lek
        qh[2*k] = qh[2*k] + qh1*alpha[j]
        qh[2*k+1] = qh[2*k+1] + qh2*alpha[j]

for i in range(N+L):
    for k in range(2*NE):
        d[i] = d[i] - G[i,k]*qh[k] 

for k in range(N):
    for j in range(N+L):
        r = math.sqrt((xedge[k]-xnode[j])**2 + (yedge[k]-ynode[j])**2)
        uh1 = (r**3)/9 + (r**2)/4
        uh[k] = uh[k] + uh1*alpha[j]

for i in range(N+L):
    for k in range(N):
        d[i] = d[i] + H[i,k]*uh[k]  
        
for i in range(N,N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[i]-xnode[j])**2 + (ynode[i]-ynode[j])**2)
        uh1 = (r**3)/9 + (r**2)/4
        d[i] = d[i] + uh1*alpha[j]
        
system_freeterm[0:N] = system_freeterm[0:N] + d[0:N]
q = np.linalg.solve(system_matrix,system_freeterm)

#daca sunt conditii mixte, aici ar trebui sa faci sortarea

rez = np.zeros(L)

for i in range(L):
    for k in range(N-1):
        rez[i] = rez[i] + (G[N+i,2*k]+G[N+i,2*k+1])*q[k+1]
    rez[i] = rez[i] + (G[N+i,0] + G[N+i,2*NE-1])*q[0]
    for k in range(N):
        rez[i] = rez[i] - H[N+i,k]*u[k]
    rez[i] = rez[i] + d[N+i]
    
plt.plot(xnode[N:N+L],rez)
plt.show()