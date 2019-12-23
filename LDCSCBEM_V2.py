# -*- coding: utf-8 -*-
"""
Created on Wed Oct 2 11:27:51 2019

@author: Florin Albisoru
"""

#import area
import numpy as np
import matplotlib.pyplot as plt
import math

import time
start_time = time.time()

#declaration area

length = 0.025
N = int(1/length)
n = 4 * N
sing = 0
m = 80
coef = 0.5

#edges of the BE
xedge = np.zeros(n + 1)
yedge = np.zeros(n + 1)

#midpoints of BE
xmid = np.zeros(n)
ymid = np.zeros(n)

#boundary conditions 
wD = np.zeros(n)
wN = np.zeros(n)
psiD = np.zeros(n)
psiN = np.zeros(n)

#matrices A,B,C,D
A = np.zeros((n,n))
B = np.zeros((n,n))
C = np.zeros((n,n))
D = np.zeros((n,n))

#system for finding the BC
system_freeterm = np.zeros(2*n)
system_unknown = np.zeros(2*n)
system_matrix = np.zeros((2*n,2*n))

#interior points
xint = coef*np.ones(m)
yint = np.linspace(0,1,num = m)
psiint = np.zeros(m)

#set the edges of the BE
xedge[0:N+1] = np.linspace(0,1,num=N+1)
xedge[N+1:2*N] = np.ones(N-1)
xedge[2*N:3*N+1] = np.linspace(1,0,num=N+1)

yedge[N:2*N+1] = np.linspace(0,1,num=N+1)
yedge[2*N+1:3*N] = np.ones(N-1)
yedge[3*N:4*N+1] = np.linspace(1,0,num=N+1)

#set the midpoints of the BE
for i in range(0,n):
    xmid[i] = (xedge[i]+xedge[i+1])/2
    ymid[i] = (yedge[i]+yedge[i+1])/2
    
#set the boundary conditions for psi and w at the midpoints (where we know them)
psiN[2*N:3*N] = -1*np.ones(N)

#compute the matrix A
def OffDiagA(x0,y0,x1,y1,x2,y2):
    dy1 = y1 - y0;
    dx1 = x1 - x0;
    dy2 = y2 - y0;
    dx2 = x2 - x0;
    dl1 = math.sqrt(dx1**2+dy1**2);
    cos1 = dx1/dl1;
    sin1 = dy1/dl1;
    dx2r = dx2*cos1 + dy2*sin1;
    dy2r = -dx2*sin1 + dy2*cos1;
    xi = math.atan2(dy2r,dx2r);
    return xi;

def FormA():
    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                A[i,j] = OffDiagA(xmid[i],ymid[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            else:
                A[i,j] =  math.pi

#compute the matrix B
def OffDiagB(x0,y0,x1,y1,x2,y2):
    a = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    b = math.sqrt((x2-x0)**2 + (y2-y0)**2)
    xi = OffDiagA(x0,y0,x1,y1,x2,y2)
    cosbeta = (a**2 + length**2 - b**2)/(2*a*length) 
    sinbeta = (b*math.sin(xi))/length;
    minusI = - (a*(math.log(a)-math.log(b))*cosbeta + length*math.log(b) - length + a*xi*sinbeta)
    return minusI            

def FormB():
    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                B[i,j] = OffDiagB(xmid[i],ymid[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            else:
                B[i,j] = - length*(math.log(length/2) - 1)
            
#compute the matrix C
def OffDiagC(x0,y0,x1,y1,x2,y2):
    a = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    b = math.sqrt((x2-x0)**2 + (y2-y0)**2)
    xi = OffDiagA(x0,y0,x1,y1,x2,y2)
    sinbeta = (b*math.sin(xi))/length
    I = - OffDiagB(x0,y0,x1,y1,x2,y2)
    c = - 0.25*a*(2*I - length)*sinbeta
    return c

def FormC():
    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                C[i,j] = OffDiagC(xmid[i],ymid[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])

#compute the matrix D
def OffDiagD(x0,y0,x1,y1,x2,y2):
    a = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    b = math.sqrt((x2-x0)**2 + (y2-y0)**2)
    xi = OffDiagA(x0,y0,x1,y1,x2,y2);
    cosbeta = (a**2 + length**2 - b**2)/(2*a*length) 
    sinbeta = (b*math.sin(xi))/length
    part1 = ((length - cosbeta)**3)*(math.log(b) - (4/3)) + ((a*cosbeta)**3)*(math.log(a) - (4/3))
    part2 = ((a*sinbeta)**2)*(- OffDiagB(x0,y0,x1,y1,x2,y2) - (2/3)*length - (1/3)*a*xi*sinbeta)
    d = 0.25 * ((1/3)*part1 + part2)
    return d

def FormD():
    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                D[i,j] = OffDiagD(xmid[i],ymid[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
            else:
                D[i,j] = ((length**3)*(math.log(length/2) - (4/3)))/48

#algorithm for Gaussian elimination with backward substitution
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

FormA()
FormB()
FormC()
FormD()

system_freeterm[0:n] = - np.dot(B,psiN)
system_matrix[0:n,0:n] = C
system_matrix[0:n,n:2*n] = D
system_matrix[n:2*n,0:n] = A
system_matrix[n:2*n,n:2*n] = B

system_unknown = Gauss(system_matrix,system_freeterm,2*n,sing)
wD = system_unknown[0:n]
wN = system_unknown[n:2*n]

#find the desired interior value of the stream function
for i in range(1,m-1):
    rezfinal = 0
    for j in range(n):
        rezB = OffDiagB(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
        rezC = OffDiagC(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
        rezD = OffDiagD(xint[i],yint[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])
        rezfinal = rezfinal + psiN[j]*rezB + wD[j]*rezC + wN[j]*rezD
    psiint[i] = rezfinal/(2*math.pi)
    

fig, ax = plt.subplots()
ax.plot(yint, psiint);

ax.set(xlabel='y', ylabel='stream function',
   title='Variation of the streamfunction on the line x = {:0.2f}'.format(coef))
ax.grid()

fig.savefig("testv2.png")
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

