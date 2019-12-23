# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:03:19 2019

@author: Florin Albisoru

Description : problema simpla cu elemente liniare
"""

import numpy as np
import math

N = 20
L = 1

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)

xnode[0:int(N/4)] = np.linspace(0,1,num = int(N/4),endpoint = False)
xnode[int(N/4):int(N/2)] = np.ones(int(N/4))
xnode[int(N/2):int(3*N/4)] = np.linspace(1,0, num = int(N/4), endpoint = False)

ynode[int(N/4):int(N/2)] = np.linspace(0,1,num = int(N/4),endpoint = False)
ynode[int(N/2):int(3*N/4)] = np.ones(int(N/4))
ynode[int(3*N/4):N] = np.linspace(1,0, num = int(N/4), endpoint = False)

xnode[N] = 0.5
ynode[N] = 0.5

#conditiile pe frontiera
#cu tot cu indexul 1 pt dirichlet si 2 pentru neumann
index = np.ones(N)
q = np.zeros(N)
u = np.zeros(N)
u = xnode[0:N] * ynode[0:N]

#cuadraturi Gauss
csi = np.array([0.339981,-0.339981,0.861136,-0.861136])
wg = np.array([0.652145,0.652145,0.347855,0.347855])

H = np.zeros((N+L,N))
G = np.zeros((N+L,2*N))

for i in range(N+L):
    for j in range(N):
        if j != N-1:
            k = j+1
        else:
            k = 0
        lj = math.sqrt((xnode[j]-xnode[k])**2 + (ynode[j]-ynode[k])**2)
        difx = xnode[k]-xnode[j]
        dify = ynode[k]-ynode[j]
        eta1 = dify/lj
        eta2 = -difx/lj
        if (i!=j and i!=k):
            h1 = 0
            h2 = 0
            g1 = 0
            g2 = 0
            for l in range(4):
                xcsi = xnode[j]+(1+csi[l])*(xnode[k]-xnode[j])/2 
                ycsi = ynode[j]+(1+csi[l])*(ynode[k]-ynode[j])/2
                r = math.sqrt((xnode[i]-xcsi)**2 + (ynode[i]-ycsi)**2)
                rdn = (xnode[i]-xcsi)*eta1 + (ynode[i]-ycsi)*eta2
                g1 = g1 + math.log(1/r)*1/2*(1-csi[l])*wg[l]
                g2 = g2 + math.log(1/r)*1/2*(1+csi[l])*wg[l]
                h1 = h1 + 1/(r**2)*rdn*1/2*(1-csi[l])*wg[l]
                h2 = h2 + 1/(r**2)*rdn*1/2*(1+csi[l])*wg[l]
            G[i,2*j] = g1*lj/(4*math.pi)
            G[i,2*j+1] = g2*lj/(4*math.pi)
            H[i,j] = H[i,j] - lj*h1/(4*math.pi)
            H[i,k] = H[i,k] - lj*h2/(4*math.pi)
        if(j==i):           
            G[i,2*j] = lj*(3/2 - math.log(lj))/(4*math.pi)
        if(k==i):
            G[i,2*j+1] = lj*(3/2 - math.log(lj))/(4*math.pi)
    if (i < N):
        H[i,i] = 1 - sum(H[i,j] for j in range(N))

system_freeterm = -np.matmul(H[0:N,0:N],u)
system_matrix = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if j != N-1:
            k = j+1
        else:
            k = 0 
        system_matrix[i,j] = system_matrix[i,j] - G[i,2*j]
        system_matrix[i,k] = system_matrix[i,k] - G[i,2*j+1]

q = np.linalg.solve(system_matrix,system_freeterm)

rez = 0
for k in range(0,N-1):
    rez = rez + (G[N,2*k] + G[N,2*k+1])*q[k+1]
rez = rez + (G[N,0]+G[N,2*N-1])*q[0]
for k in range(N):
    rez = rez - H[N,k]*u[k]