# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:13:13 2019

@author: Florin Albisoru

Description : polish again
"""

import numpy as np
import math
import matplotlib.pyplot as plt

no_edges = 20

xedge = np.zeros(no_edges+1)
yedge = np.zeros(no_edges+1)

xedge[0:5] = np.linspace(0,0.03,num = 5, endpoint = False)
xedge[5:10] = 0.03*np.ones(5)
xedge[10:15] = np.linspace(0.03,0,num = 5, endpoint = False)
yedge[5:10] = np.linspace(0,0.03,num = 5, endpoint = False)
yedge[10:15] = 0.03*np.ones(5)
yedge[15:20] = np.linspace(0.03,0,num = 5, endpoint = False)

no_bd_nodes = no_edges

xnode = np.zeros(no_bd_nodes)
ynode = np.zeros(no_bd_nodes)

xnode[0:no_bd_nodes] = (xedge[0:no_edges]+xedge[1:no_edges+1])/2
ynode[0:no_bd_nodes] = (yedge[0:no_edges]+yedge[1:no_edges+1])/2

no_int_nodes = 25

xintnode = np.zeros(no_int_nodes)
yintnode = np.zeros(no_int_nodes)

xintnode[0:5] = xnode[0:5]
xintnode[5:10] = xnode[0:5]
xintnode[10:15] = xnode[0:5]
xintnode[15:20] = xnode[0:5]
xintnode[20:25] = xnode[0:5]

yintnode[0:5] = ynode[5]*np.ones(5)
yintnode[5:10] = ynode[6]*np.ones(5)
yintnode[10:15] = ynode[7]*np.ones(5)
yintnode[15:20] = ynode[8]*np.ones(5)
yintnode[20:25] = ynode[9]*np.ones(5)

"""
#code to check whether the points are correct written
plt.plot(xedge, yedge, 'ro', xnode, ynode, 'bo', xintnode[0:11], yintnode[0:11], 'go')
plt.axis([0, 0.05, 0, 0.05])
plt.show()
"""

# prepare and populate the known and unknown vectors with boundary conditions

index = np.zeros(no_bd_nodes)
index[15:20] = np.ones(5)
u_dirichlet = np.zeros(no_bd_nodes)
u_neumann = np.zeros(no_bd_nodes)

u_dirichlet[0:15] = 100*np.ones(15)
u_neumann[15:20] = -10**4*np.ones(5)

# merge the nodes in a single array

xall = np.zeros(no_bd_nodes + no_int_nodes)
yall = np.zeros(no_bd_nodes + no_int_nodes)

xall[0:no_bd_nodes] = xnode
xall[no_bd_nodes:no_bd_nodes+no_int_nodes] = xintnode
yall[0:no_bd_nodes] = ynode
yall[no_bd_nodes:no_bd_nodes+no_int_nodes] = yintnode

no_all = no_bd_nodes + no_int_nodes

Q = np.zeros(no_all)
Q[0:no_all] = 10**7*(xall[0:no_all]**2 + yall[0:no_all]**2)

f = np.zeros((no_all,no_all))
for s in range(no_all):
    for k in range(no_all):
        r = math.sqrt((xall[k]-xall[s])**2 + (yall[k]-yall[s])**2)
        f[s,k] = - (1 + r)
        
a = np.linalg.solve(f,Q)

G = np.zeros((no_bd_nodes,no_bd_nodes))
H = np.zeros((no_bd_nodes,no_bd_nodes))

def OffDiag(xp,yp,x1,y1,x2,y2):
    csi = np.array([0.86113631,-0.86113631,0.33998104,-0.33998104])
    ome = np.array([0.34785485,0.34785485,0.65214515,0.65214515])
    xco = np.zeros(4)
    yco = np.zeros(4)
    ax = (x2-x1)/2
    bx = (x2+x1)/2
    ay = (y2-y1)/2
    by = (y2+y1)/2
    sl = math.sqrt(ax**2 + ay**2)
    eta1 = ay/sl
    eta2 = -ax/sl
    g = 0
    h = 0
    for i in range(4):
        xco[i] = ax*csi[i]+bx
        yco[i] = ay*csi[i]+by
        ra = math.sqrt((xp-xco[i])**2 + (yp-yco[i])**2)
        rd1 = (xco[i] - xp)/ra
        rd2 = (yco[i] - yp)/ra
        rdn = rd1*eta1 + rd2*eta2
        g = g + math.log(1/ra)*ome[i]*sl
        h = h - rdn*ome[i]*sl/ra
    g = g/(2*math.pi)
    h = h/(2*math.pi)
    return [g,h]

def DiagG(x1,y1,x2,y2):
    ax = (x2-x1)/2
    ay = (y2-y1)/2
    sr = math.sqrt(ax**2 + ay**2)
    g = 2*sr*(1 - math.log(sr))
    g = g/(2*math.pi)
    return g

for i in range(no_bd_nodes):
    for j in range(no_bd_nodes):
        if i!=j:
            G[i,j] = OffDiag(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[0]
            H[i,j] = OffDiag(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[1]
        else:
            G[i,j] = DiagG(xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            H[i,j] = 1/2
            
u = np.zeros((no_bd_nodes,no_all))
w = np.zeros((no_bd_nodes,no_all))

for j in range(no_bd_nodes):
    for k in range(no_all):
        r = math.sqrt((xall[k]-xnode[j])**2 + (yall[k]-ynode[j])**2)
        u[j,k] = (r**2)/4 +(r**3)/9
        ax = (xedge[j+1]-xedge[j])
        ay = (yedge[j+1]-yedge[j])
        sl = math.sqrt(ax**2 + ay**2)
        eta1 = ay/sl
        eta2 = -ax/sl
        w[j,k] = -(0.5+r/3)*((xall[k]-xnode[j])*eta1 + (yall[k]-ynode[j])*eta2)
        
freeterm = np.matmul(np.matmul(G,w),a) - np.matmul(np.matmul(H,u),a)

system_matrix = np.zeros((no_bd_nodes,no_bd_nodes))

for i in range(no_bd_nodes):
    if index[i]==0:
        freeterm = freeterm + H[:,i]*u_dirichlet[i]
        system_matrix[:,i] = G[:,i] 
    else:
        freeterm = freeterm - G[:,i]*u_neumann[i]
        system_matrix[:,i] = -H[:,i]
        
system_unknown = np.linalg.solve(system_matrix,freeterm)

u_dirichlet[15:20] = system_unknown[15:20]
u_neumann[0:15] = system_unknown[0:15]

H_star = np.zeros((no_int_nodes,no_bd_nodes))
G_star = np.zeros((no_int_nodes,no_bd_nodes))
u_star = np.zeros((no_int_nodes,no_all))

for i in range(no_int_nodes):
    for j in range(no_bd_nodes):
        G_star[i,j] = OffDiag(xintnode[i],yintnode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[0]
        H_star[i,j] = OffDiag(xintnode[i],yintnode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[1]

for i in range(no_int_nodes):
    for k in range(no_all):
        r = math.sqrt((xall[k]-xintnode[i])**2 + (yall[k]-yintnode[i])**2)
        u_star[i,k] = (r**2)/4 +(r**3)/9

final = np.zeros(no_int_nodes)

for i in range(no_int_nodes):
    sum1 = sum([H_star[i,j]*u_dirichlet[j] for j in range(no_bd_nodes)])
    sum2 = sum([G_star[i,j]*u_neumann[j] for j in range(no_bd_nodes)])
    sum3 = sum([a[k]*u_star[i,k] for k in range(no_all)])
    sum4 = sum([a[k]*sum([G_star[i,j]*w[j,k] for j in range(no_bd_nodes)]) for k in range(no_all)])
    sum5 = sum([a[k]*sum([H_star[i,j]*u[j,k] for j in range(no_bd_nodes)]) for k in range(no_all)])
    final[i] = sum1 - sum2 + sum3 + sum4 - sum5