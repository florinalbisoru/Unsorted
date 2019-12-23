# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 02:30:54 2019

@author: Florin Albisoru

Description : diffusion problem on a rectangle
"""

import math
import numpy as np
import matplotlib.pyplot as plt

N = 80
L = 1

dt = 0.05
T = 300 #time-steps

a = 11.234*(10**(-5))

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)] = np.linspace(0,0.3,num=int(N/4),endpoint = False)
xedge[int(N/4):int(N/2)] = 0.3*np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(0.3,0,num=int(N/4),endpoint = False)

yedge[int(N/4):int(N/2)] = np.linspace(0,0.4,num=int(N/4),endpoint = False)
yedge[int(N/2):int(3*N/4)] = 0.4*np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(0.4,0,num=int(N/4),endpoint = False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)
xnode[0:N] = (xedge[0:N] + xedge[1:N+1])/2
ynode[0:N] = (yedge[0:N] + yedge[1:N+1])/2
xnode[N] = 0.1
ynode[N] = 0.05

u = np.zeros(N+L)
q = np.zeros(N+L)

#punem conditiile pe frontiera

u[0:int(N/4)] = 40*np.ones(int(N/4))
u[int(N/2):int(3*N/4)] = 10*np.ones(int(N/4))

#cuadraturi Gauss
csi = np.array([0.339981044,-0.339981044,0.861136312,-0.861136312])
wg = np.array([0.652145155,0.652145155,0.347854845,0.347854845])

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

F = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        F[k,j] = 1+r
        
Finv = np.linalg.inv(F)

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
        
C = np.matmul(H,uh) - np.matmul(G,qh)
C = np.matmul(C,Finv)
C = -(1/a)*C

# set up vector of initial values
up = np.zeros(N+L)
qp = np.zeros(N+L)

A =((2/dt)*C + H)
B =((2/dt)*C - H)

system_matrix = np.zeros((N+L,N+L))
rezu = np.zeros(T)

for t in range(T):
    #compute the rhs
    system_freeterm = B 
    system_freeterm = np.matmul(system_freeterm,up)
    #compute the lhs and do the switch
    system_matrix[0:N+L,0:N] = G[0:N+L,0:N]
    system_matrix[0:N+L,N:N+L] = A[0:N+L,N:N+L]
    system_freeterm = system_freeterm + np.matmul(A[0:N+L,0:N],u[0:N])
    rez = np.linalg.solve(system_matrix,system_freeterm)
    q[0:N] = rez[0:N]
    u[N:N+L] = rez[N:N+L]
    up = u
    qp = q
    rezu[t] = u[N]
