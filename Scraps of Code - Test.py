# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:31:25 2019

@author: Florin Albisoru

This file is used to check certain lines of code that do not act right.
"""

# Note: Interchange of rows is functioning properly

import numpy as np;

"""
E = np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[0,0,0,0]]);
F = np.array([4,1,-3,4]);

E = np.c_[E, F];
H = E[0,0:4]

E1 = E[1,:]
E2 = E[2,:]

minE1 = np.min(E[1,:][np.nonzero(E[1,:])])
poz = np.where(E1 == -1)[0][0]

minE2 = np.min(E[2,:][np.nonzero(E[2,:])])
pozE2 = np.where(E2 == minE2)[0][0]

E3 = E[3,:]
minE3 = np.min(E[3,:][np.nonzero(E[3,:])])
pozE3 = np.where(E3 == minE3)

if (pozE3 == 0):
    print('nu!')
else:
    pozE3 = pozE3[0][0];
        

E = np.c_[E, F]
"""

#this chunk completely solves a triangle analythically given 3 points

