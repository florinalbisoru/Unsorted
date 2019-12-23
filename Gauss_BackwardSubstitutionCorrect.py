# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:25:05 2019

@author: Florin Albisoru

Gaussian Elimination with Backward Substitution as presented
in Burden & Faires.
"""

import numpy as np;

E = np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]]);
F = np.array([4,1,-3,4]);
n = len(E[0]);
sol = np.zeros(n);

G = np.array([[1,1,1],[2,2,1],[1,1,2]]);
H = np.array([4,6,6]);

def Gauss(A,B,n):
    x = np.zeros(n);
    A = np.c_[A, B]; 
    for i in range(0,n-1):
        arr = A[i:n,i];
        if np.count_nonzero(arr) == 0:
            print('no unique solution exists');
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
        print('no unique solution exists');
        return None;
    x[n-1] = A[n-1][n]/A[n-1][n-1];
    for i in range(n-2,-1,-1):
        x[i] = (A[i][n] - sum([A[i][j]*x[j] for j in range(i+1,n)]))/A[i][i]
    return x
            
sol = Gauss(E,F,n);
boolea = np.allclose(np.dot(E, sol), F)

alpha = Gauss(G,H,len(G[0]));