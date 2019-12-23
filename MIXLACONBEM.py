# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:05:47 2019

@author: Florin Albisoru

Description : This is a solution proposed to the problem in the ch.2 of the 
book of Brebbia and Dominguez at
http://www.boundaryelements.com/images/stories/openaccess/bemintro
/Chapter%202%20pages%2053-160.pdf, pages 67 - 69

edit : 17:40 it is working
"""

import numpy as np
import math

x = np.array([0,2,4,6,6,6,6,4,2,0,0,0,0])
y = np.array([0,0,0,0,2,4,6,6,6,6,4,2,0])

N = x.size - 1

xm = np.zeros(N)
ym = np.zeros(N)

for i in range(N):
    xm[i] = (x[i]+x[i+1])/2
    ym[i] = (y[i]+y[i+1])/2
    
G = np.zeros((N,N))
H = np.zeros((N,N))

cx = np.array([2,2,3,4,4])
cy = np.array([2,4,3,2,4])

L = cx.size

index = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
fi = np.zeros(N)
dfi = np.zeros(N)

for i in range(N):
    fi[i] = xm[i]*ym[i]

system_matrix = np.zeros((N,N))
system_freeterm = np.zeros(N)

pot = np.zeros(L)

sing = 0

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

def PopulateGH():
    for i in range(N):
        for j in range(N):
            if i!=j:
                G[i,j] = OffDiag(xm[i],ym[i],x[j],y[j],x[j+1],y[j+1])[0]
                H[i,j] = OffDiag(xm[i],ym[i],x[j],y[j],x[j+1],y[j+1])[1]
            else:
                G[i,j] = DiagG(x[j],y[j],x[j+1],y[j+1])
                H[i,j] = math.pi

def FormSystem():
    for i in range(N):
        if index[i] == 0:
            system_matrix[0:N,i] = - G[0:N,i]
            system_freeterm[0:N] = system_freeterm[0:N] - fi[i]*H[0:N,i]
        else:
            system_matrix[0:N,i] = H[0:N,i]
            system_freeterm[0:N] = system_freeterm[0:N] + fi[i]*G[0:N,i]
            
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
    dfi[0:N] = Gauss(system_matrix,system_freeterm,N,sing)[0:N]
    if sing == 0:
        print('System was solved regularly!')
    else:
        print('System is singular!')
        
def Rearrange():
    for i in range(N):
        if index[i] == 1:
            ch = fi[i]
            fi[i] = dfi[i]
            dfi[i] = ch
            
def Values():
    for k in range(L):
        for j in range(N):
            a = OffDiag(cx[k],cy[k],x[j],y[j],x[j+1],y[j+1])[0]
            b = OffDiag(cx[k],cy[k],x[j],y[j],x[j+1],y[j+1])[1]
            pot[k] = pot[k] + dfi[j]*a - fi[j]*b
        pot[k] = pot[k]/(2*math.pi)
        
PopulateGH()
FormSystem()
SolveSystem()
Rearrange()
Values()