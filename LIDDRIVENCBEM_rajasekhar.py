# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:38:41 2019

@author: Florin Albisoru

Desciption: problema lid driven a lu' Raja Sekhar
dar executata cu quadraturi Gauss
"""
from mpl_toolkits import mplot3d

import numpy as np
import math
import matplotlib.pyplot as plt

n = int(1/0.0625) 
#temp = np.linspace(0,1,num = n,endpoint = False)

N = 4*n
L = 50

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:n] = np.linspace(0,1,num = n,endpoint = False)
xedge[n:2*n] = np.ones(n)
xedge[2*n:3*n] = np.linspace(1,0,num = n,endpoint = False)

yedge[n:2*n] = np.linspace(0,1,num = n,endpoint = False)
yedge[2*n:3*n] = np.ones(n)
yedge[3*n:4*n] = np.linspace(1,0,num = n,endpoint = False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)

xnode[0:N] = (xedge[0:N] + xedge[1:N+1])/2
ynode[0:N] = (yedge[0:N] + yedge[1:N+1])/2
xnode[N:N+L] = 0.1*np.ones(L)
ynode[N:N+L] = np.linspace(0.01,0.99,num = L)

#conditiile pe frontiera
pd = np.zeros(N)
pn = np.zeros(N)
wd = np.zeros(N)
wn = np.zeros(N)

pn[2*n:3*n] = -1*np.ones(n)

#cuadraturi Gauss
csi = np.array([0.339981044,-0.339981044,0.861136312,-0.861136312])
wg = np.array([0.652145155,0.652145155,0.347854845,0.347854845])

#constructia matricilor
A = np.zeros((N+L,N))
B = np.zeros((N+L,N))
C = np.zeros((N+L,N))
D = np.zeros((N+L,N))

for i in range(N+L):
    for j in range(N):
        lj = math.sqrt((xedge[j+1]-xedge[j])**2 + (yedge[j+1]-yedge[j])**2)
        if i != j :
            a = 0
            b = 0
            c = 0
            d = 0
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
                b = b + math.log(r)*wg[k]
                a = a + (1/(r**2))*wg[k]*(eta1*(xcsi - xnode[i]) + eta2*(ycsi-ynode[i]))
                d = d + (r**2)*(math.log(r)-1)*wg[k]
                c = c + wg[k]*(eta1*(xcsi - xnode[i]) + eta2*(ycsi-ynode[i]))*(2*math.log(r)-1)
            A[i,j] = lj*a/2
            B[i,j] = -lj*b/2
            C[i,j] = (-1/4)*c*lj/2
            D[i,j] = (1/4)*d*lj/2
        else:
            A[i,j] = -math.pi
            B[i,j] = -lj*(math.log(lj)/2 - 1)
            C[i,j] = 0
            D[i,j] = lj*(math.log(lj/2) - (4/3))/48
            
system_freeterm = np.zeros(2*N)
system_matrix = np.zeros((2*N,2*N))

system_freeterm[0:N] = -np.matmul(A[0:N,0:N],pd)-np.matmul(B[0:N,0:N],pn)
system_matrix[0:N,0:N] = C[0:N,0:N]
system_matrix[0:N,N:2*N] = D[0:N,0:N]
system_matrix[N:2*N,0:N] = A[0:N,0:N]
system_matrix[N:2*N,N:2*N] = B[0:N,0:N]

sol = np.linalg.solve(system_matrix,system_freeterm)
wd = sol[0:N]
wn = sol[N:2*N]

psi = np.zeros(L)
for i in range(L):
    psi[i] = np.dot(A[N+i,:],pd) + np.dot(B[N+i,:],pn) + np.dot(C[N+i,:],wd) + np.dot(D[N+i,:],wn)
    psi[i] = psi[i]/(2*math.pi)

plt.plot(ynode[N:N+L],psi)
plt.show()   

