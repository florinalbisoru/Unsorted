# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:45 2019

@author: Florin Albisoru

Note: This program is inspired from the LABECON program of
John T. Katsikadelis, portraied in the book : John T. Katsikadelis - 
The Boundary Element Method for Engineers and Scientists. 
Theory and Applications-Academic Press (2016) and the work of M.A. Kelmanson -
An Integral Equation Method for the Solution of Singular Flow Problems, Journal
of Computational Physics, 51, 139-158, 1983.

Description : This program solves the following equation:
    L(u) = 0 in \Omega = E(a,b)
    du/dn = 2*(b^2*x^2 - a^2*y^2)/math.sqrt(b^4*x^2 + a^4*y^2) on \partial Omega,
(where L denotes the Laplacian) by the const. element BEM. This problem has the
exact solution u = x^2 - y^2 + C (without loss of generality assume C = 0).

Description of the variables :
    N = number of the boundary elements.
    m = number of internal points where the solution is computed.
    a = length of E's semi-major axis.
    b = length of E's semi-minor axis.
    xedge = array containing the x-coordinate of the edge points of all BE.
    yedge = like-wise.
    xnode = array containing the x-coordinate of the mid points of all BE.
    ynode = like-wise.
    xint = array containing the x-coordinate of the interior points
    yint = like-wise.
    uint = array containing the values of the function u at the interior points.
    uD = array containing the values of the function u on the boundary at the
         mid points of all BE.
    uN = array containing the values of du/dn on the boundary at the mid 
         points of all BE.
    H = matrix that contains the integral over each BE of the fundam. sol..
    G = matrix that contains the integral over each BE of the normal deriv.
        of the fundamental solution.

"""

#Import zone
import numpy as np;
import matplotlib.pyplot as plt;
import math;

#Declarations 
N = 20;
m = 12;
a = 5;
b = 3;

xedge = np.zeros(N+1);
yedge = np.zeros(N+1);
xnode = np.zeros(N);
ynode = np.zeros(N);

xint = np.zeros(m);
yint = np.zeros(m);
uint = np.zeros(m);

uD = np.zeros(N);
uN = np.zeros(N);

G = np.zeros((N,N));
H = np.zeros((N,N));

system_unknown = np.zeros(N);

sing = 0;

#procedure that creates our edges, nodes, interior points and the boundary conditions
def FormData():
    theta = np.zeros(N);
    dtheta = 2*math.pi/N;
    for i in range(0,N):
        theta[i] = dtheta*i;
        xedge[i] = a*math.cos(theta[i]);
        yedge[i] = b*math.sin(theta[i]);
    xedge[N]=xedge[0];
    yedge[N]=yedge[0];
    #set the middle points inside the boundary elements
    for i in range(N):
        xnode[i] = (xedge[i]+xedge[i+1])/2;
        ynode[i] = (yedge[i]+yedge[i+1])/2;
    #set the boundary condition according to our studied problem
        uN[i] = 2*(b**2*xnode[i]**2 - a**2*ynode[i]**2)/math.sqrt(b**4*xnode[i]**2 + a**4*ynode[i]**2);
    uN[N-1] = 0;
    uD[N-1] = xnode[N-1]**2 - ynode[N-1]**2;
    theta2 = np.zeros(m);
    dtheta2 = 2*math.pi/m;
    for i in range(0,m):
        theta2[i] = dtheta2*i;
        xint[i] = (a/2)*math.cos(theta2[i]);
        yint[i] = (b/2)*math.sin(theta2[i]);
        
#procedure to compute the off-diagonal elements of the matrix G
def AngleZeta(x0,y0,x1,y1,x2,y2):
    a = math.sqrt((y1-y0)**2 + (x1-x0)**2); 
    b = math.sqrt((y2-y0)**2 + (x2-x0)**2); 
    length = math.sqrt((y2-y1)**2 + (x2-x1)**2); 
    coszeta = (a**2 + b**2 - length**2)/(2*a*b);
    rez = math.acos(coszeta);
    return rez;

#procedure to form the matrix G
def FormG():
    for i in range(0,N):
        for j in range(0,N):
            if i!=j:
                G[i,j] = 1/(2*math.pi)*AngleZeta(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                G[i,j] = - 0.5;


#procedure to compute the off-diagonal elements of the matrix H
def OffDiagH(x0,y0,x1,y1,x2,y2):
    a = math.sqrt((y1-y0)**2 + (x1-x0)**2); 
    b = math.sqrt((y2-y0)**2 + (x2-x0)**2); 
    length = math.sqrt((y2-y1)**2 + (x2-x1)**2);
    coszeta = (a**2 + b**2 - length**2)/(2*a*b);
    zeta = math.acos(coszeta);
    cosbeta = (a**2 + length**2 - b**2)/(2*a*length);
    sinbeta = (b*math.sin(zeta))/length;
    rez = a*(math.log(a) - math.log(b))*cosbeta + length*math.log(b) - length + a*zeta*sinbeta;
    return rez;

#procedure to form the matrix H
def FormH():
    for i in range(0,N):
        for j in range(0,N):
            if i != j:
                H[i,j] = - OffDiagH(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1])/(2*math.pi);
            else:
                length = math.sqrt((yedge[j+1]-yedge[j])**2 + (xedge[j+1]-xedge[j])**2);
                H[i,j] = -length*(math.log(length/2) - 1)/(2*math.pi);

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

def FormSolveSystem():
    #recall that only the last condition is a Dirichlet one and all the others
    #are Neumann so the matrices that appear in the system are different;
    S = np.zeros((N,N));
    F = np.zeros(N);
    S[:,0:N-1] = G[:,0:N-1];
    S[:,N-1] = H[:,N-1];
    F = - np.dot(H,uN) - uD[N-1]*(G[:,N-1]);
    #solve the obtained system
    rez = Gauss(S,F,N,sing);
    if sing == 0:
        print("\n The system has been solved regularly! \n");
    else:
        print("\n The system is singular! \n");
    return rez;

#we put back into place the required values of the boundary conditions
def SetBoundaryValues():
    uD[0:N-1] = system_unknown[0:N-1];
    uN[N-1] = system_unknown[N-1];
    
#we compute the values of u in the interior
def UInterior():
    for k in range(0,m):
        for j in range(0,N):
            rezG = AngleZeta(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1])/(2*math.pi);
            rezH = -OffDiagH(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1])/(2*math.pi);
            uint[k] = uint[k] + rezG*uD[j] + rezH*uN[j];
    
FormData();
FormG();
FormH();

system_unknown = FormSolveSystem();

SetBoundaryValues();
UInterior();
