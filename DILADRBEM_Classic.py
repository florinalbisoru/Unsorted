# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:33:01 2019

@author: Florin Albisoru

Try again to solve Delta u = - 1 on [0,1]^2, u = 0 on Gamma with DRBEM
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#---------------------------Function Area--------------------------------------

#function to compute the off-diagonal elements of G
def RegLinIntConstElem(x0,y0,x1,y1,x2,y2):
    #ra = dist. of point O form the Gauss integration point on the BE
    #wg = weights of Gauss integration
    #xi = the coordinates of the Gauss integration in [-1,1]
    #xc, yc = the global coordinates of the Gauss integration points
    
    xc = np.zeros(4);
    yc = np.zeros(4);
    
    xi = [-0.86113631,-0.33998104,0.33998104,0.86113631];
    wg = [0.34785485,0.65214515,0.65214515,0.34785485];
    ax = (x2-x1)/2;
    ay = (y2-y1)/2;
    bx = (x2+x1)/2;
    by = (y2+y1)/2;

    # Compute the integral
    rez = 0;
    for i in range(4):
        xc[i] = ax*xi[i]+bx;
        yc[i] = ay*xi[i]+by;
        ra = math.sqrt((xc[i]-x0)**2 + (yc[i]-y0)**2);
        rez = rez + math.log(ra)*wg[i];
    sl = 2.0*math.sqrt(ax**2 + ay**2);
    rez = rez*sl/(4.0*math.pi);
    return rez;

#function to compute the diagonal elements of G
def SingLinIntConstElem(x1,y1,x2,y2):
    ax = (x2-x1)/2;
    ay = (y2-y1)/2; 
    sl = math.sqrt(ax**2 + ay**2);
    rez = sl*(math.log(sl)-1.0)/math.pi;
    return rez;

#procedure that creates the matrix G
def FormG():
    for i in range(N):
        for j in range(N):
            if i != j:
                G[i][j] = RegLinIntConstElem(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                G[i][j] = SingLinIntConstElem(xedge[j],yedge[j],xedge[j+1],yedge[j+1]);

#function to compute the off-diagonal elements of H             
def DAlpha(x0,y0,x1,y1,x2,y2):
    dy1 = y1 - y0;
    dx1 = x1 - x0;
    dy2 = y2 - y0;
    dx2 = x2 - x0;
    dl1 = math.sqrt(dx1**2+dy1**2);
    cos1 = dx1/dl1;
    sin1 = dy1/dl1;
    dx2r = dx2*cos1 + dy2*sin1;
    dy2r = -dx2*sin1 + dy2*cos1;
    da = math.atan2(dy2r,dx2r);
    rez = da/(2.0*math.pi);
    return rez;

#procedure that creates the matrix H
def FormH():
    for i in range(N):
        for j in range(N):
            if i!=j:
                H[i][j] = DAlpha(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                H[i][j] = -0.5;

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

#--------------------------End of Function Area--------------------------------

#auxiliaries
temp = np.linspace(0,1,num = 5)
sing = 0

#number of boundary elements
N = (temp.size-1)*4
L = 10

#creation of edges of boundary elements
xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

#population of edges
xedge[0:int(N/4)] = temp[0:int(N/4)]
xedge[int(N/4):int(N/2)] = np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(1,0,num=(temp.size-1),endpoint=False)

yedge[int(N/4):int(N/2)] = temp[0:int(N/4)]
yedge[int(N/2):int(3*N/4)] = np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(1,0,num=(temp.size-1),endpoint=False)

#creation of the nodes
xnode = np.zeros(N)
ynode = np.zeros(N)

#population of the nodes
for i in range(N):
    xnode[i]=(xedge[i]+xedge[i+1])/2
    ynode[i]=(yedge[i]+yedge[i+1])/2

#interior points
xint = np.linspace(0.1,0.9,num = L)
yint = 0.5*np.ones(L)

#put the nodes and interior points all together
xall = np.zeros(N+L)
yall = np.zeros(N+L)
xall[0:N] = xnode
xall[N:(N+L)] = xint
yall[0:N] = ynode
yall[N:(N+L)] = yint

#view the edges, nodes and internal points
plt.plot(xedge,yedge,'ro')
plt.plot(xnode,ynode,'bo')
plt.plot(xint,yint,'go')
plt.axis([0,2,0,2])
plt.show

#create and put the known boundary conditions
uD = np.zeros(N) # known to be zero
uN = np.zeros(N) # to be determined

#we need the matrices G and H for our method
#G - integral of the fundamental solution of the Laplacian
#H - integral of the normal derivative of the f.s. of the Laplacian
G = np.zeros((N,N))
H = np.zeros((N,N))

#call the formation of the matrices
FormG();
FormH();

#--------------Solving [Q] = [f][a] for [a]------------------------------------

Q = np.ones(N+L)
f = np.zeros((N+L,N+L))

for s in range(N+L):
    for k in range(N+L):
        dist = math.sqrt((xall[k]-xall[s])**2 + (yall[k]-yall[s])**2)
        f[s,k] = 1 + dist
        
a = Gauss(f,Q,N+L,sing)
if sing == 0:
    print("\n The system has been solved regularly! \n");
else:
    print("\n The system is singular! \n");

#------------------------------------------------------------------------------

#-------------Finding uhat and unhat ------------------------------------------
    
uhat = np.zeros((N,N+L))
unhat = np.zeros((N,N+L))

for j in range(N):
    for k in range(N+L):
        dist = math.sqrt((xall[k]-xnode[j])**2 + (xall[k]-xnode[j])**2)
        uhat[j,k] = (dist**2)/4 + (dist**3)/9
        
for j in range(N):
    length = math.sqrt((xedge[j+1] - xedge[j])**2 + (yedge[j+1] - yedge[j])**2)
    for k in range(N+L):
        dist1 = math.sqrt((xedge[j] - xall[k])**2 + (yedge[j] - yall[k])**2)
        dist2 = math.sqrt((xedge[j+1] - xall[k])**2 + (yedge[j+1] - yall[k])**2)
        unhat1 = (0.5 + dist1/3)*((xedge[j]-xall[k])*(yedge[j]-yedge[j+1]) + (yedge[j]-yall[k])*(xedge[j+1]-xedge[j]))/length
        unhat2 = (0.5 + dist2/3)*((xedge[j+1]-xall[k])*(yedge[j]-yedge[j+1]) + (yedge[j+1]-yall[k])*(xedge[j+1]-xedge[j]))/length
        unhat[j,k] = unhat1
        
#------------------------------------------------------------------------------
        
#-------------Solve System Gun = (Gunhat - Huhat)a-----------------------------

system_matrix = G
system_freeterm = np.dot((np.dot(G,unhat) - np.dot(H,uhat)),a)
system_unknown = np.zeros(N)

system_unknown = Gauss(system_matrix,system_freeterm,N,sing)
if sing == 0:
    print("\n The system has been solved regularly! \n");
else:
    print("\n The system is singular! \n");
uN = system_unknown

#------------------------------------------------------------------------------

#---------------Calculate u interior ------------------------------------------

uint = np.zeros(L)
for i in range(L):
    for j in range(N):
        rezH = DAlpha(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
        rezG = RegLinIntConstElem(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
        uint[i] = uint[i] + rezH*uD[j] - rezG*uN[j]
    for k in range(N+L):
        uint[i] = uint[i] + a[k]*uhat[i,k]
    rez = 0
    for k in range(N+L):
        for j in range(N):
            rezH = DAlpha(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            rezG = RegLinIntConstElem(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            rez = rez + rezG*unhat[j,k] - rezH*uhat[j,k]
        uint[i] = uint[i] + a[k]*rez
        
