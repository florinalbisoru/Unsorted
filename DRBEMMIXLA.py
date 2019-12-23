# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:22:31 2019

@author: Florin Albisoru

This programs solves with DRBEM the content in the file 2005_1_art_19
in the case of the Q(x,y) = 10^7*(x^2 + y^2)
"""

#import area
import numpy as np
import math
import matplotlib.pyplot as plt

#declaration area
N = 20;
L = 25;
sing = 0

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)
xnode = np.zeros(N)
ynode = np.zeros(N)
xall = np.zeros(N+L) 
yall = np.zeros(N+L)
xint = np.zeros(L)
yint = np.zeros(L)
uint = np.zeros(L)

xedge[0:int(N/4)+1] = np.linspace(0,0.03,num=int(N/4)+1)
xedge[int(N/4)+1:int(N/2)] = 0.03*np.ones(int(N/4)-1)
xedge[int(N/2):int(3*N/4)+1] = np.linspace(0.03,0,num=int(N/4)+1)

yedge[int(N/4):int(N/2)+1] = np.linspace(0,0.03,num=int(N/4)+1)
yedge[int(N/2)+1:int(3*N/4)] = 0.03*np.ones(int(N/4)-1)
yedge[int(3*N/4):N+1] = np.linspace(0.03,0,num=int(N/4)+1)

for i in range(N):
    xnode[i] = (xedge[i]+xedge[i+1])/2
    ynode[i] = (yedge[i]+yedge[i+1])/2

xint[0:int(L/5)] = xnode[0:int(L/5)]
xint[int(L/5):int(2*L/5)] = xnode[0:int(L/5)]
xint[int(2*L/5):int(3*L/5)] = xnode[0:int(L/5)]
xint[int(3*L/5):int(4*L/5)] = xnode[0:int(L/5)]
xint[int(4*L/5):L] = xnode[0:int(L/5)]

yint[0:int(L/5)] = xnode[0]*np.ones(int(L/5))
yint[int(L/5):int(2*L/5)] = xnode[1]*np.ones(int(L/5))
yint[int(2*L/5):int(3*L/5)] = xnode[2]*np.ones(int(L/5))
yint[int(3*L/5):int(4*L/5)] = xnode[3]*np.ones(int(L/5))
yint[int(4*L/5):L] = xnode[4]*np.ones(int(L/5))

xall[0:N] = xnode[0:N]
xall[N:N+L] = xint[0:L]

yall[0:N] = ynode[0:N]
yall[N:N+L] = yint[0:L]

uD = np.zeros(N)
uD[0:3*int(N/4)] = 100*np.ones(3*int(N/4))
uN = np.zeros(N)
uN[3*int(N/4):N] = 10**(-4)*np.ones(int(N/4))

f = np.zeros((N+L,N+L))
Q = np.zeros(N+L)
a = np.zeros(N+L)

uhat = np.zeros((N,N+L))
unhat = np.zeros((N,N+L))

nx = np.zeros(N);
ny = np.zeros(N);

nx[int(N/4):int(N/2)] = np.ones(int(N/4)) 
nx[int(3*N/4):N] = -1*np.ones(int(N/4))
ny[0:int(N/4)] = -1*np.ones(int(N/4))
ny[int(N/2):int(3*N/4)] = np.ones(int(N/4)) 

G = np.zeros((N,N));
H = np.zeros((N,N));

computable_rhs = np.zeros(N)

system_matrix = np.zeros((N,N))
temporary_matrix = np.zeros((N,N))
system_freeterm = np.zeros(N)
temporary_term = np.zeros(N)
system_unknown = np.zeros(N)

#procedure to compute the distance between two points
def Dist(x0,y0,x1,y1):
    return math.sqrt((x1-x0)**2 + (y1-y0)**2)

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

for s in range(N+L):
    Q[s] = -(10**7)*(xall[s]**2 + yall[s]**2)
    for k in range(N+L):
        f[s,k] = - (1 + Dist(xall[s],yall[s],xall[k],yall[k]))
        
a = Gauss(f,Q,N+L,sing)

if sing == 0:
        print("\n The system has been solved regularly! \n");
else:
        print("\n The system is singular! \n");
        
for j in range(N):
    for k in range(N+L):
        d = Dist(xnode[j],ynode[j],xall[k],yall[k])
        uhat[j,k] = (d**2/4) + (d**3/9)
        unhat[j,k] = - (-(xall[k]-xnode[j])*nx[j] + (yall[k]-ynode[j])*ny[j])*((1/2)+(1/3)*d)
        
FormG()
FormH()

computable_rhs = np.dot(np.dot(G,unhat) - np.dot(H,uhat),a)

temporary_matrix[:,0:int(3*N/4)] = H[:,0:int(3*N/4)] 
temporary_matrix[:,int(3*N/4):N] = - G[:,int(3*N/4):N]
temporary_term[0:int(3*N/4)] = uD[0:int(3*N/4)]
temporary_term[int(3*N/4):N] = uN[int(3*N/4):N]

system_freeterm = computable_rhs + np.dot(temporary_matrix,temporary_term)

system_matrix[:,0:int(3*N/4)] = G[:,0:int(3*N/4)] 
system_matrix[:,int(3*N/4):N] = - H[:,int(3*N/4):N]

system_unknown = Gauss(system_matrix,system_freeterm,N,sing)
if sing == 0:
        print("\n The system has been solved regularly! \n");
else:
        print("\n The system is singular! \n");
        
uD[int(3*N/4):N] = system_unknown[int(3*N/4):N]
uN[0:int(3*N/4)] = system_unknown[0:int(3*N/4)]

#procedure to compute u at the internal points
def UInterior():
    #Now, we compute the other values of uint.
    for i in range(0,L):
        for j in range(0,N):
            rezH = DAlpha(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            rezG = RegLinIntConstElem(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            uint[i] = uint[i] + rezH*uD[j] - rezG*uN[j];
        for k in range(N+L):
            rez = 0;
            d = Dist(xint[i],yint[j],xall[k],yall[k])
            uik = (d**2/4) + (d**3/9)
            for j in range(0,N):
                rezH = DAlpha(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
                rezG = RegLinIntConstElem(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
                rez = rez + rezG*unhat[j,k] - rezH*uhat[j,k]
            uint[i] = uint[i] + a[k]*(uik + rez)
            
UInterior()