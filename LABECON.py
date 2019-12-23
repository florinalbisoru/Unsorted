# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:07:52 2019

@author: Florin Albisoru

Python Implementation of the Code in the book of
John T. Katsikadelis - The Boundary Element Method
for Engineers and Scientists. Theory and Applications
- Academic Press 2016.

This program solves the two dimensional Laplace equation
using the Boundary Element Method with Constant boundary
elements.
"""

"""
Description of variables:
    n = N - nr. of boundary elements and boundary nodes
    m = IN - nr. of internal points where the solution is computed
    index_verify = INDEX - one-dimensional array in which a type of bc is assigned to
            the nodes. 0 when u is prescribed, 1 when du/dn is prescribed
    xl - x-coordinates of the extreme points of all elements
    yl - y -||-
    xm - x-coordinates of all the boundary nodes
    ym - y -||-
    G - matrice
    H - matrice
    A - matrice
    ub - one-dim array; at input it contains the bv of u if INDEX(j)=0
         or du/dn if INDEX(j)=1; at output it contains all the boundary
         nodal values of u.
    unb - one-dim array; it contains the RHS vector of equ. AX=B. At
          output it contains the boundary nodal values of du/du.
    xin - one-dim array that contain the x-coord of the internal points
          at which the values of u are computed
    yin - -||- y-coord -||-
    uin - one-dim array that contains the computed values of u at the internal
          points.
"""

"""
IMPORTS AREA
"""
import numpy as np
import math

"""
Parameters
"""

n = 16
m = 9
index_verify = np.zeros(n)
xl = np.zeros(n+1)
yl = np.zeros(n+1)
xm = np.zeros(n)
ym = np.zeros(n)
G = np.zeros((n,n))
H = np.zeros((n,n))
ub = np.zeros(n)
A = np.zeros((n,n))
unb = np.zeros(n)
xin = np.zeros(m)
yin = np.zeros(m)
uin = np.zeros(m)
sing = 0;

"""
Reads the data from the input file.
"""
def Input():
    i = 0;
    j = 0;
    file_input = open('Input.txt','r');
    file_input_int = open('Input_InteriorPoints.txt','r');
    for line in file_input:
        fields = line.split(',');
        xl[i] = fields[0];
        yl[i] = fields[1];
        index_verify[i] = fields[2];
        ub[i] = fields[3];
        i = i+1;
    for line in file_input_int:
        fields = line.split(',');
        xin[j] = fields[0];
        yin[j] = fields[1];
        j = j+1;
    #setting the last point on the boundary to be the first
    xl[n]=xl[0];
    yl[n]=yl[0];
    file_input.close();
    file_input_int.close();

"""
Subroutine to compute the off-line diagonal elements of the G matrix.
Regular Line Integral for Constant Elements
"""
def RLINTC(x0,y0,x1,y1,x2,y2):
    """
    ra = dist. of point O form the Gauss integration point on the BE
    wg = weights of Gauss integration
    xi = the coordinates of the Gauss integration in [-1,1]
    xc, yc = the global coordinates of the Gauss integration points
    """
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

"""
Subroutine to compute the diagonal elements of the matrix G.
Singular Line Integral for Constant Elements
"""    
def SLINTC(x1,y1,x2,y2):
    ax = (x2-x1)/2;
    ay = (y2-y1)/2; 
    sl = math.sqrt(ax**2 + ay**2);
    rez = sl*(math.log(sl)-1.0)/math.pi;
    return rez;

"""
Forms the matrix G.
"""
def GMatrix():
    #Compute the nodal coordinates and distribute them in arrays
    for i in range(n):
        xm[i] = (xl[i]+xl[i+1])/2;
        ym[i] = (yl[i]+yl[i+1])/2;
    for i in range(n):
        for j in range(n):
            if i != j:
                G[i][j] = RLINTC(xm[i],ym[i],xl[j],yl[j],xl[j+1],yl[j+1]);
            else:
                G[i][j] = SLINTC(xl[j],yl[j],xl[j+1],yl[j+1]);

"""
Subroutine to compute the off-diagonal elements of the matrix H
"""
def DALPHA(x0,y0,x1,y1,x2,y2):
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
"""
Forms the matrix H.
"""
def HMatrix():
    for i in range(n):
        xm[i] = (xl[i]+xl[i+1])/2;
        ym[i] = (yl[i]+yl[i+1])/2;
    for i in range(n):
        for j in range(n):
            if i!=j:
                H[i][j] = DALPHA(xm[i],ym[i],xl[j],yl[j],xl[j+1],yl[j+1]);
            else:
                H[i][j] = -0.5;
    
"""
Rearrange the matrices G,H according to the 
boundary conditions and forms the matrices A and B
"""
def ABMatrices():
    #Reorder the columns of the system of equations and
#store them in A
    for j in range(n):
        if index_verify[j] == 0:
            for i in range(n):
                A[i][j]=-G[i][j];
        else:
            for i in range(n):
                A[i][j]=H[i][j];
#Compute the RHS vector and store it in unb
    for i in range(n):
        for j in range(n):
            if (index_verify[j] == 0):
                unb[i] = unb[i] - H[i][j]*ub[j];
            else:
                unb[i] = unb[i] + G[i][j]*ub[j];

"""
This subroutine uses Gaussian elimination to solve
a system of linear equations AX=B.
A = 1-dim array that contains the occasional row
    of the 2-dim array of coeff of the unknowns
B = 1-dim array which contains the known coefficients
sing = 0 is the sistem has been solved regularly
       1 elsewise.
"""
def Gauss(A,B,n,sing):
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
    B[n-1] = A[n-1][n]/A[n-1][n-1];
    for i in range(n-2,-1,-1):
        B[i] = (A[i][n] - sum([A[i][j]*B[j] for j in range(i+1,n)]))/A[i][i]
"""
Solves the system of linear equations AX=B using 
Gaussian elimination
"""
def SolveSystem():
    Gauss(A,unb,n,sing);
    if sing == 0:
        print("\n The system has been solved regularly! \n");
    else:
        print("\n The system is singular! \n");

"""
Rearrange the boundary values and forms the vectors u and un
such that all the values of u are stored in ub and all the
values of the normal derivative are stored in unb
"""
def Reorder():
    for i in range(0,n):
        if index_verify[i]:
            ch = ub[i];
            ub[i] = unb[i];
            unb[i] = ch;
    
"""
Computes the values of u at the internal points
"""
def UInterior():
    for k in range(0,m):
        for j in range(0,n):
            rezH = DALPHA(xin[k],yin[k],xl[j],yl[j],xl[j+1],yl[j+1]);
            rezG = RLINTC(xin[k],yin[k],xl[j],yl[j],xl[j+1],yl[j+1]);
            uin[k] = uin[k] + rezH*ub[j] - rezG*unb[j];
    
"""
Writes the results in the output file
"""
def Output():
    file_output = open('Output.txt','w');
    file_output.write('RESULTS \n \n');
    file_output.write('Boundary nodes: \n \n');
    file_output.write('X '+'Y'+' '+'U'+' '+'Un \n');
    for i in range(0,n):
        file_output.write(str(xm[i])+' '+str(ym[i])+' '+str(ub[i])+' '+str(unb[i])+'\n');
    file_output.write('Internal points: \n \n');
    file_output.write('X'+' '+'Y'+' '+'SOLUTION U \n')
    for i in range(0,m):
        file_output.write(str(xin[i])+' '+str(yin[i])+' '+str(uin[i])+'\n');
    file_output.close();
    
Input();
GMatrix();
HMatrix();
ABMatrices();
SolveSystem();
Reorder();
UInterior();
Output();