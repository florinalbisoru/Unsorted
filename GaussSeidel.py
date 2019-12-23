# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:44:22 2019

@author: Florin Albisoru

Gauss-Seidel iterative method as presented
in Burden & Faires.
"""

import numpy as np;

n = 4;
eps = 0.00001;
x0 = np.zeros(n);
N = 10;
xsol = np.zeros(n);

A = np.array([[10,-1,2,0],[-1,11,-1,3],[2,-1,10,-1],[0,3,-1,8]]);
b = np.array([6,25,-11,15]);

n1 = 4;
eps1 = 1e-8;
x01 = np.zeros(n1);
N1 = 1000;
xsol1 = np.zeros(n1);

A1 = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])
b1 = np.array([6., 25., -11., 15.])

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
        
xsol = GaussSeidel(A,b,n,x0,eps,N);
#GaussSeidel(A1,b1,n1,x01,eps1,N1,xsol1);
xsol2 = np.linalg.solve(A,b);
