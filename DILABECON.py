# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:43:18 2019

@author: Florin Albisoru

Note: This program is inspired from the LABECON program of
John T. Katsikadelis, portraied in the book : John T. Katsikadelis - 
The Boundary Element Method for Engineers and Scientists. 
Theory and Applications-Academic Press (2016).

Description :  This program solves the following equation:
    L(u) = -1 in \Omega = [0,1] X [0,1]
    u = 0 on \partial Omega,
(where L denotes the Laplacian) by the method of transformation to the 
Laplace Equation.
              The solution is sought in the form: u = u0+u1, where u1 is a 
particular solution of the equation
    L(u1) = -1.
For this purpose we have chosen u1 = -(x^2)/2.
              It remains to find u0, which is the solution to the following
homogeneous, elliptic bvp:
    L(u0) = 0 in \Omega
    u0 = (x^2)/2 on \partial Omega.
              We solve the homogeneous problem with the following code.
              Let u0:=u and let ufinal:=u+u1.
              
Description of the variables :
    n - number of the boundary elements.
    m - number of internal points where the solution is computed.
    xedge - array containing the x-coordinate of the edge points of all BE.
    yedge - like-wise.
    xnode - array containing the x-coordinate of the mid points of all BE.
    ynode - like-wise.
    xint - array containing the x-coordinate of the interior points
    yint - like-wise.
    uint - array containing the values of the function u at the interior points.
    uD - array containing the values of the function u on the boundary at the
         mid points of all BE.
    uN - array containing the values of du/dn on the boundary at the mid 
         points of all BE.
    G - matrix that contains the integral over each BE of the fundam. sol..
    H - matrix that contains the integral over each BE of the normal deriv.
        of the fundamental solution.
"""

#Import zone
import numpy as np;
import matplotlib.pyplot as plt;
import math;

#Declarations 
n = 16;
m = 20;

xedge = np.zeros(n+1);
yedge = np.zeros(n+1);
xnode = np.zeros(n);
ynode = np.zeros(n);
xint = np.linspace(0,1,m);
yint = 0.5*np.ones(m);
uint = np.zeros(m);
ufinal = np.zeros(m);

uD = np.zeros(n);
uN = np.zeros(n);

G = np.zeros((n,n));
H = np.zeros((n,n));

sing = 0;

def InputBoundaryPoints():
   file_input = open('input.txt','r');
   i=0;
   for line in file_input:
        fields = line.split(',');
        xedge[i] = fields[0];
        yedge[i] = fields[1];
        i = i+1;
    #setting the last point to be the first for the last BE
   xedge[n]=xedge[0];
   yedge[n]=yedge[0];
   file_input.close(); 
   
def PrepareNodesAndCondition():
    #set the middle points inside the boundary elements
    for i in range(n):
        xnode[i] = (xedge[i]+xedge[i+1])/2;
        ynode[i] = (yedge[i]+yedge[i+1])/2;
    #set the boundary condition according to our studied problem
        uD[i] = (xnode[i]**2)/2;

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
    for i in range(n):
        for j in range(n):
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
    for i in range(n):
        for j in range(n):
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
    uN[n-1] = A[n-1][n]/A[n-1][n-1];
    for i in range(n-2,-1,-1):
        uN[i] = (A[i][n] - sum([A[i][j]*uN[j] for j in range(i+1,n)]))/A[i][i];
    return x;

#procedure to solve the sistem of linear equation
def SolveSystem():
    b = H.dot(uD);
    Gauss(G,b,n,sing);
    if sing == 0:
        print("\n The system has been solved regularly! \n");
    else:
        print("\n The system is singular! \n");

#procedure to compute u at the internal points
def UInterior():
    #We have included in the interior points two nodes on the boundary.
    #Hence, we set their value in advance.
    uint[0] = (xint[0]**2)/2;
    uint[m-1] = (xint[m-1]**2)/2;
    #Now, we compute the other values of uint.
    for k in range(1,m-1):
        for j in range(0,n):
            rezH = DAlpha(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            rezG = RegLinIntConstElem(xint[k],yint[k],xedge[j],yedge[j],xedge[j+1],yedge[j+1]);
            uint[k] = uint[k] + rezH*uD[j] - rezG*uN[j];

#procedure to obtain the solution of the Poisson equation
def Ufinal():
    for k in range(0,m):
        ufinal[k] = uint[k] - (xint[k]**2)/2;
        
#procedure to plot the resulting "temperature" 
def PlotTemperature():
    fig, ax = plt.subplots()
    ax.plot(xint, ufinal);

    ax.set(xlabel='x', ylabel='temperature',
       title='Variation of temperature on the line y = 0.5')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    
InputBoundaryPoints();
PrepareNodesAndCondition();
FormG();
FormH();
SolveSystem();
UInterior();
Ufinal();
PlotTemperature();
