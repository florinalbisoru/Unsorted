# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:30:55 2019

@author: Florin Albisoru

Description :  This program solves the following equation:
    L(u) = -1 in \Omega = [0,1] X [0,1]
    u = 0 on \partial Omega,
(where L denotes the Laplacian) by the finite diferences method.

Description of the variables:
    xnode = the nodes in the x-direction
    ynode = similar
    nx = number of nodes in x-direction
    ny = similar
    ufinal = the results of our computation at the midpoint y = 0.5
    A,B = matrices that appear after the deduction of discretized equation
            (see details in PDF)
    
"""

#Import zone
import numpy as np;
import matplotlib.pyplot as plt;
import math;

#Declaration of variables
xnode = np.linspace(0,1,num=31);
ynode = np.linspace(0,1,num=31);
dx = dy = xnode[1] - xnode[0];
nx = xnode.size;
ny = ynode.size;
ufinal = np.zeros(ny);

#Auxiliary matrices
A = 4*(np.diag(np.ones(ny-2))) + np.diag(-1*np.ones(ny-3),1) + np.diag(-1*np.ones(ny-3),-1);
B = np.diag(-1*np.ones(ny-2));  

#Matrices for the formation of the system
System_matrix = np.zeros(((nx-2)*(ny-2),(nx-2)*(ny-2))) + np.diag(-1*np.ones((nx-2)*(ny-2)-(ny-2)),ny-2) + np.diag(-1*np.ones((nx-2)*(ny-2)-(ny-2)),-(ny-2));
System_freeterm = (dx**2)*np.ones((nx-2)*(ny-2));
System_unknown = np.zeros((nx-2)*(ny-2));

#Parameter for the Gauss Seidel method
N = 700;

def PopulateSystemMatrix():
    for i in range(0,(nx-2)*(ny-2),nx-2):
        System_matrix[i:i+(nx-2),i:i+(nx-2)] = A;
        
def GaussSeidel(A,b,n,x0,eps,N):
    for k in range(1,N+1):
        xn = np.zeros(n);
        print("Iteration {0}: {1}".format(k, x0))
        for i in range(0,n):
            xn[i] = (- A[i,:i].dot(xn[:i]) - A[i,i+1:].dot(x0[i+1:]) + b[i])/A[i][i];
        if np.allclose(x0,xn,rtol=eps):
            print('Procedure successful!');
            print('Iterations required: '+ str(k));
            return x0;
        x0 = xn;
    print('Maximum iterations exceeded! Procedure was not successful!');
    
#procedure to plot the resulting "temperature" 
def PlotTemperature():
    fig, ax = plt.subplots()
    ax.plot(xnode, ufinal);

    ax.set(xlabel='x', ylabel='temperature',
       title='Variation of temperature on the line y = 0.5')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    
PopulateSystemMatrix();
sol = GaussSeidel(System_matrix,System_freeterm,(nx-2)*(ny-2),System_unknown,0.00001,N);
for i in range(0,(nx-2)):
    ufinal[i+1] = sol[((ny-2)//2-1)+i*(ny-2)];
#PlotTemperature();
#am selectat in acest for doar acele pct pentru care y = 0.5

xdrbem = np.linspace(0.1,0.9,num=9)
rezdrbem = np.array([0.033554,0.054531,0.068502,0.076295,0.078801,0.076295,0.068502,0.054531,0.033554])

plt.plot(xnode,ufinal,color='black',linestyle='dashed')
#plt.plot(xdrbem,rezdrbem, color= 'red')

with open('fdm_results.txt','w') as f:
    for i in range(31):
        f.write(str(xnode[i])+','+str(ufinal[i])+'\n')