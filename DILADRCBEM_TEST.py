# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:27:33 2019

@author: Florin Albisoru

problema test cu -10^6/52 pe dreptunghi

PERFECT a iesit
"""
import numpy as np
import math

N = 20
L = 1

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)+1] = np.linspace(0,0.6,num = int(N/4)+1, endpoint = False)
xedge[int(N/4)+1:int(N/2)] = 0.6*np.ones(int(N/4)-1)
xedge[int(N/2):int(3*N/4)+1] = np.linspace(0.6,0, num = int(N/4)+1, endpoint = False)

yedge[int(N/4)+1:int(N/2)] = np.linspace(0,0.4,num = int(N/4)-1, endpoint = False)
yedge[int(N/2):int(3*N/4)+1] = 0.4*np.ones(int(N/4)+1)
yedge[int(3*N/4)+1:N] = np.linspace(0.4,0,num = int(N/4)-1, endpoint = False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)

xnode[0:N] = (xedge[0:N] + xedge[1:N+1])/2
xnode[N] = 0.3

ynode[0:N] = (yedge[0:N] + yedge[1:N+1])/2
ynode[N] = 0.2

#conditiile pe frontiera
q = np.zeros(N)
# se da ca u e 0 pe frontiera
u = np.zeros(N)

#termenul liber
b = ((-10**6)/52)*np.ones(N+L)
F = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        F[k,j] = 1 + r
        
alpha = np.linalg.solve(F,b)

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
            
uh = np.zeros((N+L,N+L))
qh = np.zeros((N,N+L))

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
        
system_matrix = -G[0:N,0:N]
system_freeterm = np.matmul(H[0:N,0:N],uh[0:N,:]) - np.matmul(G[0:N,0:N],qh)
system_freeterm = np.matmul(system_freeterm,alpha)

q = np.linalg.solve(system_matrix,system_freeterm)

rez = - np.dot(H[N,0:N],u[0:N]) + np.dot(G[N,0:N],q[0:N])
temp1 = np.matmul(H[N,0:N],uh[0:N,:])
rez = rez + np.dot(temp1,alpha)
temp2 = np.matmul(G[N,0:N],qh)
rez = rez - np.dot(temp2,alpha)
rez = rez + np.dot(uh[N,:],alpha)

error = abs((rez - 310.1)/310.1)