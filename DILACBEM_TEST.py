# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:07:43 2019

@author: Florin Albisoru

Description : Simple problem
"""

import numpy as np
import math

N = 40
L = 1

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)] = np.linspace(0,1,num = int(N/4),endpoint = False)
xedge[int(N/4):int(N/2)] = np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(1,0, num = int(N/4), endpoint = False)

yedge[int(N/4):int(N/2)] = np.linspace(0,1,num = int(N/4),endpoint = False)
yedge[int(N/2):int(3*N/4)] = np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(1,0, num = int(N/4), endpoint = False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)

xnode[0:N] = (xedge[0:N] + xedge[1:N+1])/2
xnode[N] = 0.5

ynode[0:N] = (yedge[0:N] + yedge[1:N+1])/2
ynode[N] = 0.5

#conditiile pe frontiera
q = np.zeros(N)
u = np.zeros(N)
u = xnode[0:N] * ynode[0:N]

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
            
system_matrix = G[0:N,0:N]
system_freeterm = np.matmul(H[0:N,0:N],u)

q = np.linalg.solve(system_matrix,system_freeterm)

rez = - np.dot(H[N,0:N],u[0:N]) + np.dot(G[N,0:N],q[0:N])

