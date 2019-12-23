# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:31:45 2019

@author: Florin Albisoru

Description : problem at example 8.2 Katsikadelis
"""

import numpy as np
import math
import matplotlib.pyplot as plt

N = 20;

x_edges = np.zeros(N+1)
y_edges = np.zeros(N+1)

x_edges[0:int(N/4)] = np.linspace(-0.3,0.3,num = int(N/4),endpoint = False)
x_edges[int(N/4):int(N/2)] = 0.3*np.ones(int(N/4))
x_edges[int(N/2):int(3*N/4)] = np.linspace(0.3,-0.3,num = int(N/4),endpoint = False)
x_edges[int(3*N/4):N] = -0.3*np.ones(int(N/4))

y_edges[0:int(N/4)] = -0.2*np.ones(int(N/4))
y_edges[int(N/4):int(N/2)] = np.linspace(-0.2,0.2,num = int(N/4),endpoint = False)
y_edges[int(N/2):int(3*N/4)] = 0.2*np.ones(int(N/4))
y_edges[int(3*N/4):N] = np.linspace(0.2,-0.2,num = int(N/4),endpoint = False)

x_edges[N]=x_edges[0]
y_edges[N]=y_edges[0]

x_nodes = (x_edges[0:N] + x_edges[1:N+1])/2 
y_nodes = (y_edges[0:N] + y_edges[1:N+1])/2

L = 25
x_int = np.zeros(L)
y_int = np.zeros(L)

for i in range(0,L,5):
    x_int[i:i+5] = x_nodes[0:5]

y_int[0:int(L/5)] = y_nodes[5]*np.ones(int(L/5))
y_int[int(L/5):int(2*L/5)] = y_nodes[6]*np.ones(int(L/5))
y_int[int(2*L/5):int(3*L/5)] = y_nodes[7]*np.ones(int(L/5))
y_int[int(3*L/5):int(4*L/5)] = y_nodes[8]*np.ones(int(L/5))
y_int[int(4*L/5):L] = y_nodes[9]*np.ones(int(L/5))

#code to check whether the points are correct written
plt.plot(x_edges, y_edges, 'ro',x_nodes,y_nodes,'bo',x_int,y_int,'go')
plt.axis([-0.5, 0.5, -0.5, 0.5])
plt.title('Discretization of the domain')
plt.show()

no_all = N+L
x_all = np.zeros(N+L)
y_all = np.zeros(N+L)
x_all[0:N] = x_nodes
x_all[N:N+L] = x_int
y_all[0:N] = y_nodes
y_all[N:N+L] = y_int


Q = (10**6)/52 * np.ones(N+L)
f = np.zeros((N+L,N+L))
for i in range(N+L):
    for j in range(N+L):
        r = math.sqrt((x_all[i] - x_all[j])**2+(y_all[i]-y_all[j])**2)
        f[i,j] = -(1 + r)
        
a = np.linalg.solve(f,Q)

u = np.zeros(N)
q = np.zeros(N)

G = np.zeros((N,N))
H = np.zeros((N,N))

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
        h = h - rdn*ome[i]*sl/ra
    return [g,h]

def DiagG(x1,y1,x2,y2):
    ax = (x2-x1)/2
    ay = (y2-y1)/2
    sr = math.sqrt(ax**2 + ay**2)
    g = 2*sr*(1 - math.log(sr))
    return g

for i in range(N):
    for j in range(N):
        if i!=j:
            G[i,j] = (1/2*math.pi)*OffDiag(x_nodes[i],y_nodes[i],x_edges[j],y_edges[j],x_edges[j+1],y_edges[j+1])[0]
            H[i,j] = (1/2*math.pi)*OffDiag(x_nodes[i],y_nodes[i],x_edges[j],y_edges[j],x_edges[j+1],y_edges[j+1])[1]
        else:
            G[i,j] = (1/2*math.pi)*DiagG(x_edges[j],y_edges[j],x_edges[j+1],y_edges[j+1])
            H[i,j] = 1/2
            
qh = np.zeros((N,no_all))
uh = np.zeros((N,no_all))

for k in range(N):
    for j in range(N+L):
        r = math.sqrt((x_nodes[k] - x_all[j])**2+(y_nodes[k]-y_all[j])**2)
        uh[k,j] = (r**2)/4 + (r**3)/9
        
for k in range(N):
    for j in range(N+L):
        r = math.sqrt((x_nodes[k] - x_all[j])**2+(y_nodes[k]-y_all[j])**2)
        ax = (x_edges[k+1] - x_edges[k])
        ay = (y_edges[k+1] - y_edges[k])
        l = math.sqrt(ax**2+ay**2)
        eta1 = ay/l
        eta2 = -ax/l
        rdn = (x_nodes[k] - x_all[j])*eta1 + (y_nodes[k]-y_all[j])*eta2
        qh[k,j] = rdn*(0.5+r/3)
        
freeterm = np.dot(np.dot(H,uh) - np.dot(G,qh),a)

q = np.linalg.solve(-G,freeterm)

H_star = np.zeros((L,N))
G_star = np.zeros((L,N))
uh_star = np.zeros((L,N+L))
final = np.zeros(L)

for i in range(L):
    for j in range(N):
        G_star[i,j] = (1/2*math.pi)*OffDiag(x_int[i],y_int[i],x_edges[j],y_edges[j],x_edges[j+1],y_edges[j+1])[0]
        H_star[i,j] = (1/2*math.pi)*OffDiag(x_int[i],y_int[i],x_edges[j],y_edges[j],x_edges[j+1],y_edges[j+1])[1]

for k in range(L):
    for j in range(N+L):
        r = math.sqrt((x_int[k] - x_all[j])**2+(y_int[k]-y_all[j])**2)
        uh_star[k,j] = (r**2)/4 + (r**3)/9
        
rhs = np.dot(np.dot(H_star,uh) - np.dot(G_star,qh),a) - np.dot(uh_star,a)
final = np.dot(G_star,q) + rhs
