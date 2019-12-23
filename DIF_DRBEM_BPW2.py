# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:55:14 2019

@author: Florin Albisoru

Description : V2

FUNCTIONEAZA!!!
"""

import math
import numpy as np
import matplotlib.pyplot as plt

N = 40
L = 33
dt = 0.05 #time step
a = 1.25 #thermal diffusivity coefficient
T = 24 #time interval right end

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)] = np.linspace(0,3,num=int(N/4),endpoint=False)
xedge[int(N/4):int(N/2)] = 3*np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(3,0,num=int(N/4),endpoint=False)

yedge[int(N/4):int(N/2)] = np.linspace(0,3,num=int(N/4),endpoint=False)
yedge[int(N/2):int(3*N/4)] = 3*np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(3,0,num=int(N/4),endpoint=False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)
xnode[0:N] = (xedge[0:N] + xedge[1:N+1])/2
ynode[0:N] = (yedge[0:N] + yedge[1:N+1])/2

xnode[N:N+6] = np.linspace(0.6,2.4,num=6,endpoint=False)
xnode[N+6:N+12] = 2.4*np.ones(6)
xnode[N+12:N+18] = np.linspace(2.4,0.6,num=6,endpoint=False)
xnode[N+18:N+24] = 0.6*np.ones(6)
xnode[N+24:N+33] = [1.2,1.5,1.8,1.8,1.8,1.5,1.2,1.2,1.5]
ynode[N:N+6] = 0.6*np.ones(6)
ynode[N+6:N+12] = np.linspace(0.6,2.4,num=6,endpoint=False)
ynode[N+12:N+18] = 2.4*np.ones(6)
ynode[N+18:N+24] = np.linspace(2.4,0.6,num=6,endpoint=False)
ynode[N+24:N+33] = [1.2,1.2,1.2,1.5,1.8,1.8,1.8,1.5,1.5]

up = 30*np.ones(N+L)
qp = np.zeros(N+L)

u = np.zeros(N+L)
q = np.zeros(N+L)

#plt.plot(xnode,ynode,'ro')

F = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        F[k,j] = 1 + r
        
Finv = np.linalg.inv(F)

#cuadraturi Gauss
csi = np.array([0.339981,-0.339981,0.861136,-0.861136])
wg = np.array([0.652145,0.652145,0.347855,0.347855])

#constructia matricilor
G = np.zeros((N+L,N+L))
H = np.zeros((N+L,N+L))

for i in range(N+L):
    for j in range(N):
        lj = math.sqrt((xedge[j+1]-xedge[j])**2 + (yedge[j+1]-yedge[j])**2)
        if i != j :
            g = 0
            h = 0
            sumx = xedge[j+1]+xedge[j]
            sumy = yedge[j+1]+yedge[j]
            difx = xedge[j+1]-xedge[j]
            dify = yedge[j+1]-yedge[j]
            eta1 = dify/lj
            eta2 = -difx/lj
            for k in range(4):
                xcsi = difx/2*csi[k] + sumx/2
                ycsi = dify/2*csi[k] + sumy/2
                r = math.sqrt((xcsi - xnode[i])**2 + (ycsi - ynode[i])**2)
                g = g + math.log(1/r)*wg[k]
                h = h + (1/(r**2))*wg[k]*(eta1*(xcsi - xnode[i]) + eta2*(ycsi-ynode[i]))
            G[i,j] = lj*g/(4*math.pi)
            H[i,j] = -lj*h/(4*math.pi)
        else:
            G[i,j] = lj*(math.log(2/lj) + 1)/(2*math.pi)
            H[i,j] = 1/2
H[N:N+L,N:N+L] = np.eye(L)  
       
          
uh = np.zeros((N+L,N+L))
qh = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        uh[k,j] = (r**2)/4 + (r**3)/9
        
for k in range(N):
    lk = math.sqrt((xedge[k+1]-xedge[k])**2 + (yedge[k+1]-yedge[k])**2)
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        difx = xedge[k+1]-xedge[k]
        dify = yedge[k+1]-yedge[k]
        eta1 = dify/lk
        eta2 = -difx/lk
        qh[k,j] = (0.5+r/3)*((xnode[k]-xnode[j])*eta1 + (ynode[k]-ynode[j])*eta2)
        
C = np.matmul(H,uh)
C = C - np.matmul(G,qh)
C = np.matmul(C,Finv)
C = -(1/a)*C

RHS = (2/dt)*C - H
LHS = (2/dt)*C + H

rezfin = np.zeros(T+1)
rezfin[0] = 30

for t in range(T):
    system_freeterm = np.zeros(N+L)
    system_matrix = np.zeros((N+L,N+L))
    rez = np.zeros(N+L)
    system_freeterm = np.matmul(RHS,up)
    system_matrix[0:N+L,0:N] = -2*G[0:N+L,0:N]
    system_matrix[0:N+L,N:N+L] = LHS[0:N+L,N:N+L]
    system_freeterm = system_freeterm - np.matmul(LHS[0:N+L,0:N],u[0:N])
    rez = np.linalg.solve(system_matrix,system_freeterm)
    q[0:N] = rez[0:N]
    u[N:N+L] = rez[N:N+L]
    up = u
    qp = q
    rezfin[t+1] =u[N+L-1]

plt.plot(np.linspace(0,T+1,num=T+1,endpoint=False),rezfin)