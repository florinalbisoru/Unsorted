# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:51:57 2019

@author: Florin Albisoru

Description : Unbiased program of ComputerProgram2 of BPW
"""

import numpy as np
import matplotlib.pyplot as plt
import math

sing = 0

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

#procedure to find the number of nodes
def InputNumbers(string):
    file = open(string,'r')
    count = len(file.readlines())
    file.close()
    return count

#procedure to read the bd and int nodes
def InputPoints(string,x,y):
    file_in = open(string,'r')
    i = 0
    for line in file_in:
        fields = line.split(',')
        x[i] = fields[0]
        y[i] = fields[1]
        i = i+1
    file_in.close()
    
#procedure to read the bd conditions
def InputCond():
    file_in = open('Input_bd_cond.txt','r')
    i = 0
    for line in file_in:
        fields = line.split(',')
        kode[i] = fields[0]
        if kode[i] == 1:
            u[i] = fields[1]
        else:
            q[i] = fields[1]
        i = i+1
    file_in.close()

def Connectivity():
    for i in range(N):
        con[i,0] = i
        con[i,1] = i+1
    con[N-1,1] = 0

def Lengths():
    for k in range(NE):
        a = x_bd_nodes[int(con[k,1])] - x_bd_nodes[int(con[k,0])]
        b = y_bd_nodes[int(con[k,1])] - y_bd_nodes[int(con[k,0])]
        le[k] = math.sqrt(a**2 + b**2)

def Alpha():
    #source term
    b = np.zeros(N+L)
    for i in range(N+L):
        b[i] = -x_all[i]
    F = np.zeros((N+L,N+L))
    for i in range(N+L):
        xi = x_all[i]
        yi = y_all[i]
        for j in range(N+L):
            xj = x_all[j]
            yj = y_all[j]
            r = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            F[i,j] = 1 + r
    a = Gauss(F,b,N+L,sing)
    return a

def Assemble():
    ni = 4
    poei = np.array([0.86113631,-0.86113631,0.33998104,-0.33998104])
    fdep = np.array([0.34785485,0.34785485,0.65214515,0.65214515])
    for j1 in range(N+L):
        xi = x_all[j1]
        yi = y_all[j1]
        cc = 0
        for j2 in range(NE):
            lj = le[j2]
            n1 = int(con[j2,0])
            n2 = int(con[j2,1])
            x1 = x_bd_nodes[n1]
            x2 = x_bd_nodes[n2]
            y1 = y_bd_nodes[n1]
            y2 = y_bd_nodes[n2]
            h1 = 0 
            h2 = 0
            g1 = 0
            g2 = 0
            if j1 != n1 and j1 != n2:
                for j3 in range(ni):
                    e = poei[j3]
                    w = fdep[j3]
                    xx = x1+(1+e)*(x2-x1)/2
                    yy = y1+(1+e)*(y2-y1)/2
                    r = math.sqrt((xi-xx)**2+(yi-yy)**2)
                    pp = w*((xi-xx)*(y1-y2)+(yi-yy)*(x2-x1))/(r*r*4*math.pi)
                    h1 = h1 + (1-e)*pp/2.0
                    h2 = h2 + (1+e)*pp/2.0
                    pp = lj*w*math.log(1/r)/(4*math.pi)
                    g1 = g1 + (1-e)*pp/2
                    g2 = g2 + (1+e)*pp/2
                cc = cc - h1 - h2
            else:
                ge = lj*(3/2 - math.log(lj))/(4*math.pi)
            if (n1 == j1): g1 = ge
            if (n2 == j1): g2 = ge
            H[j1,n1] = H[j1,n1] + h1
            H[j2,n2] = H[j1,n2] + h2
            G[j1,2*j2] = g1
            G[j1,2*j2+1] = g2
            kk = kode[n1]
            if kk==0 or kk==2:
                xy[j1] = xy[j1]+q[n1]*g1
                A[j1,n1] = A[j1,n1] + h1
            else:
                xy[j1] = xy[j1] - u[n1]*h1
                A[j1,n1] = A[j1,n1] - g1
            kk = kode[n2]
            if kk==0 or kk==2:
                xy[j1] = xy[j1] + q[n2]*g2
                A[j1,n2] = A[j1,n2] + h2
            else:
                xy[j1] = xy[j1] - u[n2]*h2
                A[j1,n2] = A[j1,n2] -g2
        H[j1,j1] = cc
        kk = kode[j1]
        if kk==0 or kk==2:
            A[j1,j1] = cc
        else:
            xy[j1] = xy[j1] - u[j1]*cc

def RHS():
    qh = np.zeros(2*NE)
    for k in range(NE):
        n1 = int(con[k,0])
        n2 = int(con[k,1])
        x1 = x_bd_nodes[n1]
        x2 = x_bd_nodes[n2]
        y1 = y_bd_nodes[n1]
        y2 = y_bd_nodes[n2]
        lek = le[k]
        k1 = 2*k 
        k2 = 2*k + 1
        for j in range(N+L):
            alphaj = alpha[j]
            xp = x_all[j]
            yp = y_all[j]
            r1 = math.sqrt((x1-xp)**2+(y1-yp)**2)
            r2 = math.sqrt((x2-xp)**2+(y2-yp)**2)
            qh1 = (0.5 + r1/3)*((x1-xp)*(y1-y2) + (y1-yp)*(x2-x1))/lek
            qh2 = (0.5 + r2/3)*((x1-xp)*(y1-y2) + (y1-yp)*(x2-x1))/lek
            qh[k1] = qh[k1] + qh1*alphaj
            qh[k2] = qh[k2] + qh2*alphaj
    for i in range(N+L):
        for k in range(2*NE):
            d[i] = d[i] - G[i,k]*qh[k]
    uh = np.zeros(N)
    for k in range(N):
        xk = x_bd_nodes[k]
        yk = y_bd_nodes[k]
        for j in range(N+L):
            alphaj = alpha[j]
            xj = x_all[j]
            yj = y_all[j]
            r = math.sqrt((xk-xj)**2+(yk-yj)**2)
            uh1 = (r**3)/9 + (r**2)/4
            uh[k] = uh[k] + uh1*alphaj
    for i in range(N+L):
        for k in range(N):
            d[i] = d[i] + H[i,k]*uh[k]
    for i in range(N,N+L):
        xi = x_all[i]
        yi = y_all[i]
        for j in range(N+L):
            alphaj = alpha[j]
            xp = x_all[j]
            yp = y_all[j]
            r = math.sqrt((xi-xp)**2 + (yi-yp)**2)
            uh1 = (r**3)/9 + (r**2)/4
            d[i] = d[i] + uh1*alphaj
            
def Interm():
    for i in range(N,N+L):
        for k in range(N-1):
            u[i] = u[i] + (G[i,2*k] + G[i,(2*k+1)])*q[k+1]
        u[i] = u[i]+(G[i,0] + G[i,2*N-1])*q[0]
        for k in range(N):
            u[i] = u[i] - H[i,k]*u[k]
        u[i] = u[i] + d[i]
            
#parameters of the program
N = InputNumbers('Input_bd_points.txt')
NE = N
L = InputNumbers('Input_int_points.txt')

#read the boundary nodes
x_bd_nodes = np.zeros(N)
y_bd_nodes = np.zeros(N)
InputPoints('Input_bd_points.txt',x_bd_nodes,y_bd_nodes)

#prepare connectivity of boundary elements
con = np.zeros((N,2))
Connectivity()

#read the interior nodes
x_int_nodes = np.zeros(L)
y_int_nodes = np.zeros(L)
InputPoints('Input_int_points.txt',x_int_nodes,y_int_nodes)

#read the boundary conditions
kode = np.zeros(N+L)
u = np.zeros(N+L)
q = np.zeros(N+L)
InputCond()

#compute the lengths
le = np.zeros(NE)
Lengths()

#put the nodes all together
x_all = np.zeros(N+L)
y_all = np.zeros(N+L)
x_all[0:N] = x_bd_nodes
x_all[N:N+L] = x_int_nodes
y_all[0:N] = y_bd_nodes
y_all[N:N+L] = y_int_nodes

#find alpha
alpha = Alpha()    

#assemble matrices H,G and prepare the system with sist. matrix A and freeterm xy
H = np.zeros((N+L,N+L))
G = np.zeros((N+L,2*N))
A = np.zeros((N+L,N+L))
xy = np.zeros(N+L)

Assemble()

#find d = (HUhat - GQhat)*alpha
d = np.zeros(N+L)
RHS()

#sum the freeterm xy and d
for i in range(N+L):
    xy[i] = xy[i] + d[i]
    
#solve the system
#xy[0:N] = np.linalg.solve(A[0:N,0:N],xy[0:N])

xy[0:N] = Gauss(A[0:N,0:N],xy[0:N],N,sing)
#put the boundary values into the appropriate array
for i in range(N):
    kk = kode[i]
    if kk==0 or kk==2:
        u[i] = xy[i]
    else:
        q[i] = xy[i]
        
Interm()

#find the potential at the internal points