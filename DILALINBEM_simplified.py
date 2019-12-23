# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:08:57 2019

@author: Florin Albisoru

Description : Reduction of the code from DILALINBEM3.py, that is 
Delta u = 0 on the unit circle, u = xy on the boundary with evaluation on the 
point 

Description : Also yields good results for Grosan's problem
"""

import math
import numpy as np
import matplotlib.pyplot as plt
"""
NN = 20 # number of boundary nodes
NE = NN # number of boundary elements
L = 1 # number of interior nodes
"""

#auxiliaries
temp = np.linspace(0,1,num = 60)

#number of boundary elements
N = (temp.size-1)*4

#---------------------------------------
NN = N# number of nodes
NE = N# number of elements
L = 15 # number of internal nodes

xnodes = np.zeros(NN+L) # x-coordinate of all the points
ynodes = np.zeros(NN+L) # y-coordinate of all the points

index = np.zeros(NN+L) #indicator for the Neumann or Dirichlet conditions
uD = np.zeros(NN+L) #Dirichlet conditions - 1 in index
uN = np.zeros(NN+L) #Neumann conditions - 2 in index

H = np.zeros((NN+L,NN+L)) #matrix H
G = np.zeros((NN+L,2*NE)) #matrix G

system_matrix = np.zeros((NN+L,NN+L))
system_freeterm = np.zeros(NN+L)

sing = 0

#generate the points
def Data():
    """
    theta = np.linspace(0,2*math.pi,num=NN,endpoint = False)
    for i in range(NN):
        xnodes[i] = math.cos(theta[i])
        ynodes[i] = math.sin(theta[i])
    xnodes[NN] = 0.5
    ynodes[NN] = 0.5
    """
    #population of nodes on the boundary
    xnodes[0:int(NN/4)] = temp[0:int(NN/4)]
    xnodes[int(NN/4):int(NN/2)] = np.ones(int(NN/4))
    xnodes[int(NN/2):int(3*NN/4)] = np.linspace(1,0,num=(temp.size-1),endpoint=False)

    ynodes[int(NN/4):int(NN/2)] = temp[0:int(NN/4)]
    ynodes[int(NN/2):int(3*NN/4)] = np.ones(int(NN/4))
    ynodes[int(3*NN/4):NN] = np.linspace(1,0,num=(temp.size-1),endpoint=False)

    #internal nodes
    xnodes[NN:NN+L] = np.linspace(0.001,0.999,num = L)
    ynodes[NN:NN+L] = 0.5*np.ones(L)

#set the boundary data as desired
def BoundaryData():
    index[0:NN] = np.ones(NN)
    for i in range(NN):
        uD[i] = (xnodes[i]**2)/2

def OffDiagG(xi,yi,x1,y1,x2,y2,dist):
    #performs gaussian integration with 4 nodes and their weights
    csi = np.array([-0.86113631,-0.33998104,0.33998104,0.86113631])
    wg = np.array([0.34785485,0.65214515,0.65214515,0.34785485])
    g1 = 0
    g2 = 0
    for k in range(4):
        xx = x1 + (1 + csi[k])*(x2 - x1)/2
        yy = y1 + (1 + csi[k])*(y2 - y1)/2
        r = math.sqrt((xi-xx)**2 + (yi-yy)**2)
        pp = dist*wg[k]*math.log(1/r)/(4*math.pi)
        g1 = g1 + (1 - csi[k])*pp/2
        g2 = g2 + (1 + csi[k])*pp/2
    return [g1,g2]
        
def OffDiagH(xi,yi,x1,y1,x2,y2,dist):
    #performs gaussian integration with 4 nodes and their weights
    csi = np.array([-0.86113631,-0.33998104,0.33998104,0.86113631])
    wg = np.array([0.34785485,0.65214515,0.65214515,0.34785485])
    h1 = 0
    h2 = 0 
    cc = 0
    for k in range(4):
        xx = x1 + (1 + csi[k])*(x2 - x1)/2
        yy = y1 + (1 + csi[k])*(y2 - y1)/2
        r = math.sqrt((xi-xx)**2 + (yi-yy)**2)
        pp = wg[k]*((xi-xx)*(y1-y2) + (yi-yy)*(x2-x1))/(r*r*4*math.pi)
        h1 = h1 + (1 - csi[k])*pp/2.0
        h2 = h2 + (1 + csi[k])*pp/2.0 
    cc = -h1 -h2
    return [h1,h2,cc]         
        
def PopulateMatricesFormSystem():
    for i in range(NN+L):
        diagH = 0
        for j in range(NE):
            h1 = 0
            h2 = 0
            g1 = 0
            g2 = 0
            if j == NE-1: k = 0 #to connect the last node with the first
            else: k = j+1
            le = math.sqrt((xnodes[k] - xnodes[j])**2 + (ynodes[k]-ynodes[j])**2)
            if i == j or i == k: 
                ge = le*(3/2 - math.log(le))/(4*math.pi)
            else:
                h1 = OffDiagH(xnodes[i],ynodes[i],xnodes[k],ynodes[k],xnodes[j],ynodes[j],le)[0]
                h2 = OffDiagH(xnodes[i],ynodes[i],xnodes[k],ynodes[k],xnodes[j],ynodes[j],le)[1]
                diagH = diagH + OffDiagH(xnodes[i],ynodes[i],xnodes[k],ynodes[k],xnodes[j],ynodes[j],le)[2]
                g1 = OffDiagG(xnodes[i],ynodes[i],xnodes[k],ynodes[k],xnodes[j],ynodes[j],le)[0]
                g2 = OffDiagG(xnodes[i],ynodes[i],xnodes[k],ynodes[k],xnodes[j],ynodes[j],le)[1]
            if j == i: g1 = ge
            if k == i: g2 = ge
            H[i,j] = H[i,j] + h1
            H[i,k] = H[i,k] + h2
            G[i,2*j] = g1
            G[i,2*j+1] = g2
            if index[j] == 2 or index[j] == 0:
                system_matrix[i,j] = system_matrix[i,j] + h1
                system_freeterm[i] = system_freeterm[i] + uN[j]*g1
            else:
                system_matrix[i,j] = system_matrix[i,j] - g1
                system_freeterm[i] = system_freeterm[i] - uD[j]*h1
            if index[k] == 2 or index[j] == 0:
                system_freeterm[i] = system_freeterm[i] + uN[k]*g2
                system_matrix[i,k] = system_matrix[i,k] + h2
            else:
                system_freeterm[i] = system_freeterm[i] - uD[k]*h2
                system_matrix[i,k] = system_matrix[i,k] - g2
        H[i,i] = diagH
        if index[i] == 2 or index[i] == 0:
            system_matrix[i,i] = diagH
        else:
            system_freeterm[i] = system_freeterm[i] - uD[i]*diagH

#implementation of Gauss elimination with backward substitution
def Gauss(A,B,n,sing):
    x = np.zeros(n);
    A = np.c_[A, B]; 
    for i in range(0,n-1):
        arr = A[i:n,i];
        if np.count_nonzero(arr) == 0:
            sing = 1;
            return None;
        minarr = np.min(arr[np.nonzero(arr)]);
        arrfull = A[0:n,i];
        p = np.where(arrfull == minarr)[0][0];
        if p != i and i <= p:
            A[[p,i],:] = A[[i,p],:];
        for j in range(i+1,n):
            m = A[j][i]/A[i][i];
            A[j,:] = A[j,:] - m * A[i,:];
    if A[n-1][n-1] == 0:
        sing = 1;
        return None;
    x[n-1] = A[n-1][n]/A[n-1][n-1];
    for i in range(n-2,-1,-1):
        x[i] = (A[i][n] - sum([A[i][j]*x[j] for j in range(i+1,n)]))/A[i][i];
    return x;

def SolveSystem():
    #system_solution=Gauss(system_matrix[0:NN,0:NN],system_freeterm[0:NN],NN,sing)
    A = system_matrix[0:NN,0:NN]
    b = system_freeterm[0:NN]
    system_solution=np.linalg.solve(A,b)
    if sing == 0:
        print('System was solved regularly!')
    else:
        print('System is singular!')
    for i in range(NN):
        k = index[i]
        if k == 0 or k == 2:
            uD[i] = system_solution[i]
        else:
            uN[i] = system_solution[i]

def UInterior():
    for i in range(NN,NN+L):
        for k in range(0,NN-1):
            uD[i] = uD[i] + (G[i,2*k] + G[i,2*k+1])*uN[k+1]
        uD[i] = uD[i] + (G[i,0] + G[i,2*NE-1])*uN[0]
        for k in range(NN):
            uD[i] = uD[i] - H[i,k]*uD[k]
        
Data()
BoundaryData()
PopulateMatricesFormSystem()
SolveSystem()
UInterior()

xfinal = xnodes[NN:NN+L]
ufinal = np.zeros(L)
for i in range(L):
    ufinal[i] = uD[NN+i] - (xnodes[NN+i]**2)/2

#procedure to plot the resulting "temperature" 
def PlotTemperature():
    fig, ax = plt.subplots()
    ax.plot(xfinal, ufinal);

    ax.set(xlabel='x', ylabel='temperature',
       title='Variation of temperature on the line y = 0.5')
    ax.grid()

    fig.savefig("testdilalinbem5_simplified.png")
    plt.show()
    
PlotTemperature()