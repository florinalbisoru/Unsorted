# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:55:04 2019

@author: Florin Albisoru

Description : Simple problem - linear elements
Update : Functioneaza
"""

import numpy as np
import math

N = 200
L = 1
NE = N

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)] = np.linspace(0,1,num = int(N/4),endpoint = False)
xedge[int(N/4):int(N/2)] = np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(1,0, num = int(N/4), endpoint = False)

yedge[int(N/4):int(N/2)] = np.linspace(0,1,num = int(N/4),endpoint = False)
yedge[int(N/2):int(3*N/4)] = np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(1,0, num = int(N/4), endpoint = False)

xnodes = np.zeros(N+L)
ynodes = np.zeros(N+L)

xnodes[0:N] = xedge[0:N]
ynodes[0:N] = yedge[0:N]
xnodes[N] = 0.5
ynodes[N] = 0.5

index = np.ones(N)
u = np.zeros(N)
q = np.zeros(N)
u[0:N] = xnodes[0:N]+ynodes[0:N]

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
                r = math.sqrt((xnodes[j1]-x)**2+(ynodes[j1]-y)**2)
                pp = -((xnodes[j1]-x)*(yedge[j2]-yedge[k]) + (ynodes[j1]-y)*(xedge[k]-xedge[j2]))/(r*r*4*math.pi)*wg[j3]
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

q = np.linalg.solve(system_matrix,system_freeterm)

#nu uita sa pui in array-ul corect rezultatele

rez = np.zeros(L)

for i in range(L):
    for k in range(N-1):
        rez[i] = rez[i] + (G[N+i,2*k]+G[N+i,2*k+1])*q[k+1]
    rez[i] = rez[i] + (G[N+i,0] + G[N+i,2*NE-1])*q[0]
    for k in range(N):
        rez[i] = rez[i] - H[N+i,k]*u[k]