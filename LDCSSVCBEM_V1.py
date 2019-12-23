# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:35:52 2019

@author: Florin Albisoru

Note: This is a program that solves the Lid-Driven Cavity problem in the case
of Stokes' system via the Stream function-Vorticity formulation with the Constant 
element - Boundary Element Method. This work is inspired by 3 bibliographical
references:
    1. Chandra Shekhar Nishad, Anirban Chandra, G.P. Raja Sekhar - 
    Stokes Flow Inside Topographically Patterned Microchannel Using Boundary 
    Element Method, International Journal of Chemical Reactor Engineering,
    doi: 10.1515/ijcre-2017-0057.
    2. John T. Katsikadelis - The Boundary Element Method for Engineers and 
    Scientists. Theory and Applications-Academic Press (2016).
    3. M.A. Kelmanson - An Integral Equation Method for the Solution of 
    Singular Slow Flow Problems, Journal of Computational Physics, 51, 139-158,
    1983.

Description: This program solves the following equations:
    \Delta(\psi) = - \omega in \Omega = [0,1] \times [0,1],
    \Delta(\omega) = 0 in \Omega,
    \psi = 0 on \partial \Omega,
    d \psi / dx = 0 on {0} x [0,1], {1} x [0,1],
    d \psi / dy = 0 on [0,1] x {0},
    d \psi / dy = - 1 on (0,1) x {1}.
using the Boundary Element Method.
Note that the initial equations:
    \Delta u - \nabla p = 0 in \Omega,
    div u = 0 in \Omega,
were converted using the stream function-vorticity.

Description of the variables:
    leng = length of the boundary elements (constant);
    n = number of considered boundary edges on a segment;
    N = number of considered boundary edges;
    m = number of internal points where the solution is computed
    xedge = the x-coordinate of an edge point of a boundary element
    yedge = the y-coordinate of an edge point of a boundary element
    xint - array containing the x-coordinate of the interior points
    yint - like-wise.
    psiD - array containing the values of the function psi on the boundary at the
         middle points of all BE.
    psiN - array containing the values of dpsi/dn on the boundary at the middle 
         points of all BE.
    wD - array containing the values of the function w on the boundary at the
         middle points of all BE.
    wN - array containing the values of dw/dn on the boundary at the middle 
         points of all BE.
    A - matrix that contains the integral of normal derivative of the fund. 
        sol. of the Laplacian over each BE.
    B - matrix that contains the integral of the fund. sol. of the Laplacian
        over each BE.
    C - matrix that contains the integral of the normal derivative of the fund.
        sol. of the biharmonic equation over each BE.
    D - matrix that contains the integral of the fund. sol. of the biharmonic
        equation over each BE..
"""

#Import zone
import numpy as np;
import matplotlib.pyplot as plt;
import math;

#Declarations

leng = 1;
n = int(1/leng);
m = 100;

#generate the boundary points
x = np.zeros(n+1) + np.arange(0,n+1)*leng;
y = np.zeros(n+1);
z = np.ones(n);

xedge = np.concatenate((x, z, x[n-1:0:-1], y));
yedge = np.concatenate((y,x[1:n+1],z,x[n-1:0:-1],np.zeros(1)));
N = xedge.size-1;

#xint = 0.5*(np.zeros(100));
xint = np.linspace(0.1, 0.1, num=m);
yint = np.linspace(0.01, 0.99, num=m);

psiint = np.zeros(m);
wint = np.zeros(m);

xnode=np.zeros(N);
ynode=np.zeros(N);

psiD = np.zeros(N);
psiN = np.zeros(N);

wD = np.zeros(N);
wN = np.zeros(N);

A = np.zeros((N,N));
B = np.zeros((N,N));
C = np.zeros((N,N));
D = np.zeros((N,N));

System_matrix = np.zeros((2*N,2*N));
System_freeterm = np.zeros(2*N);
System_unknown = np.zeros(2*N);

sing = 0;

#set the boundary condition according to our studied problem
psiN[int(N/2+1) : int(3*N/4)] = -1;

def PrepareNodes():
    #set the middle points inside the boundary elements
    for i in range(N):
        xnode[i] = (xedge[i]+xedge[i+1])/2;
        ynode[i] = (yedge[i]+yedge[i+1])/2;
        
#procedure that gives us the value of the angle (q_jp_iq_j+1) where q are the 
#boundary edges and p_i are the mid points of the boundary elements
def AngleZeta(x0,y0,x1,y1,x2,y2):
    dy1 = y1 - y0;
    dx1 = x1 - x0;
    dy2 = y2 - y0;
    dx2 = x2 - x0;
    dl1 = math.sqrt(dx1**2+dy1**2);
    cos1 = dx1/dl1;
    sin1 = dy1/dl1;
    dx2r = dx2*cos1 + dy2*sin1;
    dy2r = -dx2*sin1 + dy2*cos1;
    zeta = math.atan2(dy2r,dx2r);
    return zeta;   

def AngleZeta2(x0,y0,x1,y1,x2,y2):
    dy1 = y1 - y0;
    dx1 = x1 - x0;
    dy2 = y2 - y0;
    dx2 = x2 - x0;
    dl1 = math.sqrt(dx1**2+dy1**2);
    dl2 = math.sqrt(dx2**2+dy2**2);
    quant = (dl1**2+dl2**2-leng**2)/(2*dl1*dl2);
    rez = math.acos(quant)
    return rez;

#procedure that creates the matrix A
def FormA():
    for i in range(N):
        for j in range(N):
            if i!=j:
                A[i][j] = AngleZeta(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                A[i][j] = -math.pi;
                
#procedure that creates the off-diagonal elements of B
def OffDiagB(x0,y0,x1,y1,x2,y2):
    zeta = AngleZeta(x0,y0,x1,y1,x2,y2);
    b = math.sqrt((x2-x0)**2+(y2-y0)**2);
    a = math.sqrt((x1-x0)**2+(y1-y0)**2);
    sinbeta = (math.sin(zeta)*b)/leng;
    cosbeta = (a**2 + leng**2 - b**2)/(2*a*leng);
    rez = -(a*(math.log(a/b)*cosbeta) +leng*math.log(b) - leng + a*zeta*sinbeta);
    return rez;
                
#procedure that creates the matrix B
def FormB():
    for i in range(N):
        for j in range(N):
            if i!=j:
                B[i][j] = OffDiagB(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                B[i][j] = -leng*(math.log(leng/2) - 1);

#procedure that creates the off-diagonal elements of C
def OffDiagC(x0,y0,x1,y1,x2,y2):
    zeta = AngleZeta(x0,y0,x1,y1,x2,y2);
    dist2 = math.sqrt((x2-x0)**2+(y2-y0)**2);
    dist1 = math.sqrt((x1-x0)**2+(y1-y0)**2);
    sinbeta = (math.sin(zeta)*dist2)/leng;
    interm = OffDiagB(x0,y0,x1,y1,x2,y2);
    rez = -(dist1*(2*interm - leng)*sinbeta)/4;
    return rez;

#procedure that creates the matrix C
def FormC():
    for i in range(N):
        for j in range(N):
            if i!=j:
                C[i][j] = OffDiagC(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                C[i][j] = 0;
                
#procedure that creates the off-diaognal elements of D
def OffDiagD(x0,y0,x1,y1,x2,y2):
    zeta = AngleZeta(x0,y0,x1,y1,x2,y2);
    b = math.sqrt((x2-x0)**2+(y2-y0)**2);
    a = math.sqrt((x1-x0)**2+(y1-y0)**2);
    sinbeta = (math.sin(zeta)*b)/leng;
    cosbeta = (a**2 + leng**2 - b**2)/(2*a*leng);
    interm = OffDiagB(x0,y0,x1,y1,x2,y2);
    part1 = ((leng - a*cosbeta)**3)*(math.log(b)-(4/3)) + ((a*cosbeta)**3)*(math.log(a)-(4/3));
    part2 = ((a*sinbeta)**2)*(interm - (2/3)*leng - (1/3)*a*zeta*sinbeta);
    rez = ((1/3)*part1 + part2)/4;
    return rez
            
#procedure that creates the matrix D
def FormD():
    for i in range(N):
        for j in range(N):
            if i!=j:
                D[i][j] = OffDiagD(xnode[i],ynode[i],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            else:
                D[i][j] = (leng**3*(math.log(leng/2)-(4/3)))/48;
    
#procedure that prepares the system in order to obtain the values on the boundary
# of wD and wN
def PrepareSystem():
    F = -A.dot(psiD) - B.dot(psiN);
    System_freeterm[0:N] = F;
    System_matrix[0:N,0:N] = C;
    System_matrix[0:N,N:(2*N)] = D;
    System_matrix[N:(2*N),0:N] = A;
    System_matrix[N:(2*N),N:(2*N)] = B;

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

#procedure that solves our system and separates the values of wD and wN
def SolveSystem():
    rez = np.linalg.solve(System_matrix,System_freeterm);
    #rez2 = Gauss(System_matrix,System_freeterm,2*N,sing);
    wD[0:N] = rez[0:N];
    wN[0:N] = rez[N:(2*N)];
    
def InteriorValues():
    #compute the values of psi and w at the desired interior points
    for k in range(0,m):
        rezw = 0;
        rezpsi = 0;
        for j in range(0,N):
            rezA = AngleZeta(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            rezB = OffDiagB(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            rezC = OffDiagC(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            rezD = OffDiagD(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            rezw = rezw + rezA*wD[j] + rezB*wN[j];
            rezpsi = rezpsi + rezA*psiD[j] + rezB*psiN[j] + rezC*wD[j] + rezD*wN[j];
        wint[k] = rezw/(2*math.pi);
        psiint[k] = rezpsi/(2*math.pi);
    """
    k = 0;
    rezw = 0;
    rezpsi = 0;
    for j in range(0,N):
        rezA = AngleZeta(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezB = OffDiagB(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezC = OffDiagC(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezD = OffDiagD(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezw = rezw + rezA*wD[j] + rezB*wN[j];
        rezpsi = rezpsi + rezA*psiD[j] + rezB*psiN[j] + rezC*wD[j] + rezD*wN[j];
    wint[k] = rezw/(math.pi);
    psiint[k] = rezpsi/(math.pi);
    k = m-1;
    rezw = 0;
    rezpsi = 0;
    for j in range(0,N):
        rezA = AngleZeta(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezB = OffDiagB(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezC = OffDiagC(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezD = OffDiagD(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
        rezw = rezw + rezA*wD[j] + rezB*wN[j];
        rezpsi = rezpsi + rezA*psiD[j] + rezB*psiN[j] + rezC*wD[j] + rezD*wN[j];
    wint[k] = rezw/(math.pi);
    psiint[k] = rezpsi/(math.pi);
    """
#procedure to plot the resulting "streamline" 
def PlotStreamline():
    fig, ax = plt.subplots()
    ax.plot(yint, psiint);

    ax.set(xlabel='y', ylabel='stream function',
       title='Variation of the streamfunction on the line x = 0.5')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    

PrepareNodes();

FormA();
FormB();
FormC();
FormD();
PrepareSystem();
SolveSystem();
InteriorValues();
PlotStreamline();

