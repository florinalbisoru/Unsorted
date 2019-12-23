# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:35:32 2019

@author: Florin Albisoru

Description: problema lui Grosan inca o data
"""

import numpy as np
import math
import matplotlib.pyplot as plt

no_edges = 8

xedge = np.zeros(no_edges+1)
yedge = np.zeros(no_edges+1)

xedge[0:int(no_edges/4)] = np.linspace(0,1,num = int(no_edges/4), endpoint = False)
xedge[int(no_edges/4):int(no_edges/2)] = np.ones(int(no_edges/4))
xedge[int(no_edges/2):int(3*no_edges/4)] = np.linspace(1,0,num = int(no_edges/4), endpoint = False)
yedge[int(no_edges/4):int(no_edges/2)] = np.linspace(0,1,num = int(no_edges/4), endpoint = False)
yedge[int(no_edges/2):int(3*no_edges/4)] = np.ones(int(no_edges/4))
yedge[int(3*no_edges/4):no_edges] = np.linspace(1,0,num = int(no_edges/4), endpoint = False)

no_bd_nodes = no_edges

xnode = np.zeros(no_bd_nodes)
ynode = np.zeros(no_bd_nodes)

xnode[0:no_bd_nodes] = (xedge[0:no_edges]+xedge[1:no_edges+1])/2
ynode[0:no_bd_nodes] = (yedge[0:no_edges]+yedge[1:no_edges+1])/2

no_int_nodes = 3

xintnode = np.linspace(0,1,num = no_int_nodes)
yintnode = 0.5*np.ones(no_int_nodes)

"""
#code to check whether the points are correct written
plt.plot(xedge, yedge, 'ro', xnode, ynode, 'bo', xintnode[0:25], yintnode[0:25], 'go')
plt.axis([0, 1.5, 0, 1.5])
plt.show()
"""

# on the boundary u is 0 so the unknown is u_neumann
u_dirichlet = np.zeros(no_bd_nodes)
u_neumann = np.zeros(no_bd_nodes)

xall = np.zeros(no_bd_nodes + no_int_nodes)
yall = np.zeros(no_bd_nodes + no_int_nodes)

xall[0:no_bd_nodes] = xnode
xall[no_bd_nodes:no_bd_nodes+no_int_nodes] = xintnode
yall[0:no_bd_nodes] = ynode
yall[no_bd_nodes:no_bd_nodes+no_int_nodes] = yintnode

# prepare the Q << source function >> term

no_all = no_bd_nodes + no_int_nodes

Q = -np.ones(no_all)

f = np.zeros((no_all,no_all))
for s in range(no_all):
    for k in range(no_all):
        r = math.sqrt((xall[k]-xall[s])**2 + (yall[k]-yall[s])**2)
        f[s,k] = (1 + r)
        
a = np.zeros(no_all)
a = np.linalg.solve(f,Q)

#formation of the matrices G and H

G = np.zeros((no_bd_nodes,no_bd_nodes))
H = np.zeros((no_bd_nodes,no_bd_nodes))

#procedures for the elements of G and H

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
        h = h + rdn*ome[i]*sl/ra
    return [g,h]

def DiagG(x1,y1,x2,y2):
    ax = (x2-x1)/2
    ay = (y2-y1)/2
    sr = math.sqrt(ax**2 + ay**2)
    g = 2*sr*(1 - math.log(sr))
    return g

for i in range(no_bd_nodes):
    for j in range(no_bd_nodes):
        if i!=j:
            G[i,j] = (1/2*math.pi)*OffDiag(xall[i],yall[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[0]
            H[i,j] = (1/2*math.pi)*OffDiag(xall[i],yall[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[1]
        else:
            G[i,j] = (1/2*math.pi)*DiagG(xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            H[i,j] = -1/2
            
U = np.zeros((no_bd_nodes,no_all))
W = np.zeros((no_bd_nodes,no_all))

for k in range(no_bd_nodes):
    for j in range(no_all):
        r = math.sqrt((xall[j]-xnode[k])**2 + (yall[j]-ynode[k])**2)
        U[k,j] = (r**2)/4 + (r**3)/9
    
for k in range(no_bd_nodes):  
    for j in range(no_all):      
        r = math.sqrt((xall[j]-xnode[k])**2 + (yall[j]-ynode[k])**2)
        ax = xedge[k+1] - xedge[k]
        ay = yedge[k+1] - yedge[k]
        sl = math.sqrt(ax**2 + ay**2)
        eta1 = ay/sl
        eta2 = -ax/sl
        d = (xall[j] - xnode[k])*eta1 + (yall[j]- ynode[k])*eta2
        W[k,j] = - d*(0.5+r/3)
        
freeterm = np.zeros(no_bd_nodes)
freeterm = np.dot(np.dot(G,W),a) - np.dot(np.dot(H,U),a)

# form the system
freeterm = freeterm + np.dot(H,u_dirichlet)
u_neumann = np.linalg.solve(G,freeterm)

"""
final = np.zeros(no_int_nodes)
for i in range(no_bd_nodes,no_bd_nodes+no_int_nodes):
    sum1 = sum([H[i,j]*u_dirichlet[j] for j in range(no_bd_nodes)])
    sum2 = sum([G[i,j]*u_neumann[j] for j in range(no_bd_nodes)])
    sum3 = sum([a[k]*U[i,k]for k in range(no_all)])
    sum4 = sum([a[k]*sum([G[i,j]*W[j,k] for j in range(no_bd_nodes)]) for k in range(no_all)])
    sum5 = sum([a[k]*sum([H[i,j]*U[j,k] for j in range(no_bd_nodes)]) for k in range(no_all)])
    final[i-no_bd_nodes] = sum1 - sum2 + sum3 + sum4 - sum5

final2 = np.zeros(no_int_nodes)

#procedure to plot the resulting "temperature" 
def PlotTemperature():
    fig, ax = plt.subplots()
    ax.plot(xintnode, final);

    ax.set(xlabel='x', ylabel='temperature',
       title='Variation of temperature on the line y = 0.5')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    
PlotTemperature()
"""

G_interior = np.zeros((no_int_nodes,no_bd_nodes))
H_interior = np.zeros((no_int_nodes,no_bd_nodes))
U_interior = np.zeros((no_int_nodes,no_all))

for i in range(no_int_nodes):
    for j in range(no_bd_nodes):
        H_interior[i,j] = (1/2*math.pi)*OffDiag(xall[i],yall[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[1]
        G_interior[i,j] = (1/2*math.pi)*OffDiag(xall[i],yall[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])[0]
        
for k in range(no_int_nodes):
    for j in range(no_all):
        r = math.sqrt((xall[k]-xall[j])**2 + (yall[k]-yall[j])**2)
        U_interior[k,j] = (r**2)/4 + (r**3)/9
        
final2 = np.zeros(no_int_nodes)
final2 = np.dot(H_interior,u_dirichlet) - np.dot(G_interior,u_neumann) + np.dot(U_interior,a) + np.dot(np.dot(G_interior,W),a) - np.dot(np.dot(H_interior,U),a)

def PlotTemperature():
    fig, ax = plt.subplots()
    ax.plot(xintnode, final2);

    ax.set(xlabel='x', ylabel='temperature',
       title='Variation of temperature on the line y = 0.5')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    
PlotTemperature()
