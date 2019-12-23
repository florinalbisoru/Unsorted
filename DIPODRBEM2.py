# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:38:03 2019

@author: Florin Albisoru

Description : problema lui Grosan cu DRBEM
"""

import math
import numpy as np
import matplotlib.pyplot as plt

#auxiliaries
temp = np.linspace(0,1,num = 5)

#number of boundary elements
N = (temp.size-1)*4

#---------------------------------------
NN = N# number of nodes
NE = N# number of elements
L = 9 # number of internal nodes
NI = 4

csi = np.array([0.86113631,-0.86113631,0.33998104,-0.33998104])
wg = np.array([0.34785485,0.34785485,0.65214515,0.65214515])

xnodes = np.zeros(NN+L) 
ynodes = np.zeros(NN+L)

#population of nodes on the boundary
xnodes[0:int(NN/4)] = temp[0:int(NN/4)]
xnodes[int(NN/4):int(NN/2)] = np.ones(int(NN/4))
xnodes[int(NN/2):int(3*NN/4)] = np.linspace(1,0,num=(temp.size-1),endpoint=False)

ynodes[int(NN/4):int(NN/2)] = temp[0:int(NN/4)]
ynodes[int(NN/2):int(3*NN/4)] = np.ones(int(NN/4))
ynodes[int(3*NN/4):NN] = np.linspace(1,0,num=(temp.size-1),endpoint=False)

#internal nodes
xnodes[NN:NN+L] = np.linspace(0.1,0.9,num = L)
ynodes[NN:NN+L] = 0.5*np.ones(L)

con = np.zeros((NN,2))
for i in range(NN):
    con[i,0] = i
    con[i,1] = i+1
con[NN-1,1] = 0

kode = np.zeros(NN+L)
u = np.zeros(NN+L)
q = np.zeros(NN+L)

for i in range(NN):
    kode[i] = 1
    u[i] = 0
    
leg = np.zeros(NE)
for k in range(NE):
    distx = xnodes[int(con[k,1])] - xnodes[int(con[k,0])]
    disty = ynodes[int(con[k,1])] - ynodes[int(con[k,0])]
    leg[k] = math.sqrt(distx**2 + disty**2)

b = -1*np.ones(NN)
F = np.zeros((NN,NN))

NH = NN
for i in range(NN):
    xi = xnodes[i]
    yi = ynodes[i]
    for j in range(NN):
        xj = xnodes[j]
        yj = ynodes[j]
        r = math.sqrt((xi-xj)**2 + (yi-yj)**2)
        F[i,j] = r + 1
        
alpha = np.zeros(NN+L)
alpha[0:NN] = np.linalg.solve(F,b)

H = np.zeros((NN+L,NN+L))
G = np.zeros((NN+L,2*NN))
A = np.zeros((NN+L,NN+L))
xy = np.zeros(NN+L)

#zona pentru calculul lui G si H si A si xy

for j1 in range(NN+L):
    xi = xnodes[j1]
    yi = ynodes[j1]
    cc = 0
    for j2 in range(NE):
        lj = leg[j2]
        n1 = int(con[j2,0])
        n2 = int(con[j2,1])
        x1 = xnodes[n2] #schimbat
        x2 = xnodes[n1]
        y1 = ynodes[n2]
        y2 = ynodes[n1]
        h1 = 0
        h2 = 0
        g1 = 0
        g2 = 0
        if j1 == n1 or j1 == n2:
            ge = lj*(3/2 - math.log(lj))/(4*math.pi)
        else:
           for j3 in range(NI):
               E = csi[j3]
               W = wg[j3]
               xx = x1 + (1 + E)*(x2 - x1)/2
               yy = y1 + (1 + E)*(y2 - y1)/2
               r = math.sqrt((xi-xx)**2 + (yi-yy)**2)
               pp = W*((xi-xx)*(y1-y2) + (yi-yy)*(x2-x1))/(r*r*4*math.pi)
               h1 = h1 + (1 - E)*pp/2.0
               h2 = h2 + (1 + E)*pp/2.0
               pp = lj*W*math.log(1/r)/(4*math.pi)
               g1 = g1 + (1 - E)*pp/2
               g2 = g2 + (1 + E)*pp/2
           cc = cc - h1 - h2
        if n1 == j1: g1 = ge
        if n2 == j1: g2 = ge
        H[j1,n1] = H[j1,n1] + h1
        H[j1,n2] = H[j1,n2] + h2
        G[j1,2*j2-1] = g1
        G[j1,2*j2] = g2
        kk = kode[n1]
        if kk == 0 or kk == 2:
            xy[j1] = xy[j1] + q[n1]*g1
            A[j1,n1] = A[j1,n1] + h1
        else:
            xy[j1] = xy[j1] - u[n1]*h1
            A[j1,n1] = A[j1,n1] - g1
        kk = kode[n2]
        if kk == 0 or kk == 2:
            xy[j1] = xy[j1] + q[n2]*g2
            A[j1,n2] = A[j1,n2] + h2
        else:
            xy[j1] = xy[j1] - u[n2]*h2
            A[j1,n2] = A[j1,n2] - g2
    H[j1,j1] = cc
    kk = kode[j1]
    if kk == 0 or kk == 2:
        A[j1,j1] = cc
    else:
        xy[j1] = xy[j1] - u[j1]*cc

#calculeaza d = (H Uhat - G Qhat)*alpha

d = np.zeros(NN+L)
qh = np.zeros(2*NE)
for k in range(NE):
    n1 = int(con[k,0])
    n2 = int(con[k,1])
    x1 = xnodes[n1] 
    x2 = xnodes[n2]
    y1 = ynodes[n1]
    y2 = ynodes[n2]
    lek = leg[k]
    k1 = 2*k-1
    k2 = 2*k
    for j in range(NN+L):
        alphaj = alpha[j]
        xp = xnodes[j]
        yp = ynodes[j]
        r1 = math.sqrt((x1-xp)**2 + (y1-yp)**2)
        r2 = math.sqrt((x2-xp)**2 + (y2-yp)**2)    
        qh1 = (0.5+r1/3)*((x1-xp)*(y1-y2) + (y1-yp)*(x2-x1))/lek
        qh2 = (0.5+r2/3)*((x2-xp)*(y1-y2) + (y2-yp)*(x2-x1))/lek
        qh[k1] = qh[k1] + qh1*alphaj
        qh[k2] = qh[k2] + qh2*alphaj
for i in range(NN+L):
    for k in range(2*NE):
        d[i] = d[i] - G[i,k]*qh[k]
uh = np.zeros(NN)
for k in range(NN):
    xk = xnodes[k]
    yk = ynodes[k]
    for j in range(NN+L):
        alphaj = alpha[j]
        xj = xnodes[j]
        yj = ynodes[j]
        r = math.sqrt((xk-xj)**2 + (yk-yj)**2)
        uh1 = (r**3)/9 + (r**2)/4
        uh[k] = uh[k]+uh1*alphaj
for i in range(NN+L):
    for k in range(NN):
        d[i] = d[i]+H[i,k]*uh[k]
for i in range(NN,NN+L):
    xi = xnodes[i]
    yi = ynodes[i]
    for j in range(NN+L):
        alphaj = alpha[j]
        xp = xnodes[j]
        yp = ynodes[j]
        r = math.sqrt((xi-xp)**2+(yi-yp)**2)
        uh1 = (r**3)/9 + (r**2)/4
        d[i] = d[i] + uh1*alphaj

#insumeaza xy si d

for i in range(NN):
    xy[i] = xy[i] + d[i] 
    
system_matrix = A[0:NN,0:NN]
system_freeterm = xy[0:NN]

system_solution = np.linalg.solve(system_matrix,system_freeterm)

for i in range(NN):
    kk = kode[i]
    if kk == 0 or kk == 2:
        u[i] = system_solution[i]
    else:
        q[i] = system_solution[i]

for i in range(NN,NN+L):
    for k in range(0,NN-1):
        u[i] = u[i] + (G[i,2*k] + G[i,2*k+1])*q[k+1]
    u[i] = u[i] + (G[i,0] + G[i,2*NE-1])*q[0]
    for k in range(NN):
        u[i] = u[i] - H[i,k]*u[k]