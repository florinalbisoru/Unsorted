# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:38:49 2019

@author: Florin Albisoru

Solving the problem from page 320 from Katsikadelis
"""

import numpy as np
import math
import matplotlib.pyplot as plt

tempx = np.linspace(-0.3,0.3,num = 6)
tempy = np.linspace(-0.2,0.2,num = 6)

#number of boundary edges
N = 2*tempx.size + 2*tempy.size

#creation of the boundary edges
xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

#population of the boundary edges
xedge[0:int(N/4)] = tempx
xedge[int(N/4):int(N/2)] = 0.3*np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(0.3,-0.3,num = 5)
xedge[int(3*N/4):N] = -0.3*np.ones(int(N/4))
xedge[N] = xedge[0]

yedge[0:int(N/4)] = -0.2*np.ones(int(N/4))
yedge[int(N/4):int(N/2)] = tempy
yedge[int(N/2):int(3*N/4)] = 0.2*np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(0.2,-0.2,num = 5)
yedge[N] = yedge[0]

#creation of the boundary nodes
xnode = np.zeros(N)
ynode = np.zeros(N)

#population of the boundary nodes
for i in range(N):
    xnode[i]=(xedge[i] + xedge[i+1])/2
    ynode[i]=(yedge[i] + yedge[i+1])/2

#plt.plot(xedge,yedge,'ro')
plt.plot(xnode,ynode,'bo')
plt.axis([-0.5,0.5,-0.5,0.5])
plt.show