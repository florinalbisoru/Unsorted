# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:03:10 2019

@author: Florin Albisoru

Description : incercam cu DRBEM-ul plus ceea ce se afla in fisierul MIXLACONBEM
sa rezolvam problema lui Grosan
"""


import numpy as np
import math
import matplotlib.pyplot as plt

N = 4

temp = np.linspace(0,1,num = int(N/4),endpoint = False)

x = np.zeros(N+1)
y = np.zeros(N+1)

x[0:int(N/4)] = temp
x[int(N/4):int(N/2)] = np.ones(int(N/4))
x[int(N/2):int(3*N/4)] = np.linspace(1,0,num=int(N/4),endpoint=False)

y[int(N/4):int(N/2)] = temp
y[int(N/2):int(3*N/4)] = np.ones(int(N/4))
y[int(3*N/4):N] = np.linspace(1,0,num=int(N/4),endpoint=False)

x[N] = x[0]
y[N] = y[0]

xm = np.zeros(N)
ym = np.zeros(N)

for i in range(N):
    xm[i] = (x[i]+x[i+1])/2
    ym[i] = (y[i]+y[i+1])/2
    
G = np.zeros((N,N))
H = np.zeros((N,N))

cx = np.linspace(0,1,num=5)
L = cx.size
cy = 0.5*np.ones(L)

index = np.zeros(N)
fi = np.zeros(N)
dfi = np.zeros(N)

system_matrix = np.zeros((N,N))
system_freeterm = np.zeros(N)

pot = np.zeros(L)

sing = 0

#initializarile pentru DRBEM
b = -1*np.ones(N+L)
alpha = np.ones(N+L)
f = np.zeros((N+L,N+L))

xa = np.zeros(N+L)
ya = np.zeros(N+L)

xa[0:N] = xm
xa[N:N+L] = cx
ya[0:N] = ym
ya[N:N+L] = cy

uhat = np.zeros((N,N+L))
qhat = np.zeros((N,N+L))

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

def PopulateHats():
    for k in range(N):
        lengk = math.sqrt((x[k+1]-x[k])**2+(y[k+1]-y[k])**2)
        n1 = (y[k+1] - y[k])/lengk 
        n2 = (x[k] - x[k+1])/lengk
        for j in range(N+L):
            #uhat
            r = math.sqrt((xm[k]-xa[j])**2 + (ym[k]-ya[j])**2)
            uhat[k,j] = (r**2)/4+(r**3)/9
            #qhat
            qhat[k,j] = (0.5 + r/3)*((xm[k]-xa[j])*n1+(ym[k]-ya[j])*n2)

def FindAlpha():
    for i in range(N+L):
        for j in range(N+L):
            r = math.sqrt((xa[i]-xa[j])**2 + (ya[i]-ya[j])**2)
            f[i,j] = 1+r
    alpha[0:N+L] = Gauss(f,b,N+L,sing)

def FormSystem():
    for i in range(N):
        if index[i] == 0:
            system_matrix[0:N,i] = - G[0:N,i]
            system_freeterm[0:N] = system_freeterm[0:N] - fi[i]*H[0:N,i]
        else:
            system_matrix[0:N,i] = H[0:N,i]
            system_freeterm[0:N] = system_freeterm[0:N] + fi[i]*G[0:N,i]
    d1 = np.dot(H,uhat) - np.dot(G,qhat)
    d2 = np.dot(d1,alpha)
    system_freeterm[0:N] = system_freeterm[0:N] + d2      

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
    for i in range(L):
        for j in range(N+L):
            suma = 0
            for k in range(N):
                a = OffDiag(cx[i],cy[i],x[k],y[k],x[k+1],y[k+1])[0]
                b = OffDiag(cx[i],cy[i],x[k],y[k],x[k+1],y[k+1])[1]
                suma = suma + b*uhat[k,j] - a*qhat[k,j]
            r = math.sqrt((cx[i]-xa[j])**2+(cy[i]-ya[j])**2)
            hat = (r**2)/4+(r**3)/9
            suma = suma + hat
            suma = alpha[j]*suma
            pot[i] = pot[i] + suma/(2*math.pi)
    pot[0] = pot[0]*2
    pot[L-1] = pot[L-1]*2                

PopulateGH()
FindAlpha()
PopulateHats()
FormSystem()
SolveSystem()
#Rearrange()
Values()

