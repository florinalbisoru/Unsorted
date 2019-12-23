# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:37:54 2019

@author: Program pentru generarea datelor
"""

import numpy as np

N = 20
x = np.zeros(N)
y = np.zeros(N)

x[5:11] = np.linspace(0.1,0.6,num=6)
y[0:5] = np.linspace(0,0.4,num=5)
x[11:15] =0.6*np.ones(4)
y[5:11] = 0.4*np.ones(6)
y[11:15] = np.linspace(0.3,0, num=4)
x[15:20] = np.linspace(0.5,0,num =5, endpoint = False)

f = open("newdata.txt","w")

for i in range(N):
    f.write(str(x[i])+','+str(y[i])+'\n')
    
f.close()
