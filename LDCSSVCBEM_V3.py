# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:00:46 2019

@author: Florin Albisoru

Description : LidDrivenCavityStokes as in RajaSekhar et al 2017
"""

import math
import numpy as np
import matplotlib.pyplot as plt

N = 64
temp = np.linspace(0,1,num = N,endpoint=False)

x = np.zeros(4*N + 1)
y = np.zeros(4*N + 1)

x[0:N] = temp[0:N]
x[N:2*N] = np.ones(N)
x[2*N:3*N] = np.linspace(1,0,num=N,endpoint=False)

y[N:2*N] = temp[0:N] 
y[2*N:3*N] = np.ones(N)
y[3*N:4*N] = np.linspace(1,0,num=N,endpoint=False)

x[4*N] = x[0]
y[4*N] = y[0]

xm = np.zeros(4*N)
ym = np.zeros(4*N)

NN = 4*N

for i in range(NN):
    xm[i] = (x[i]+x[i+1])/2 
    ym[i] = (y[i]+y[i+1])/2

L = 160

cx = 0.5*np.ones(L)
cy = np.linspace(0,1,num = L)

psi = np.zeros(L)

#conditiile pe frontiera
#cunoscute
pD = np.zeros(NN)
pN = np.zeros(NN)
pN[int(NN/2):int(3*NN/4)] = -1*np.ones(N)
#necunoscute
wD = np.zeros(NN)
wN = np.zeros(NN)

A = np.zeros((NN,NN))
B = np.zeros((NN,NN))
C = np.zeros((NN,NN))
D = np.zeros((NN,NN))

system_matrix = np.zeros((2*NN,2*NN))
system_freeterm = np.zeros(2*NN)
system_unknown = np.zeros(2*NN)

#pot = np.zeros(L)

sing = 0

def OffDiag(xp,yp,x1,y1,x2,y2):
    """
    temp1 = x2
    x2 = x1
    x1 = temp1
    temp2 = y2
    y2 = y1
    y1 = temp2
    """
    csi = np.array([0.86113631,-0.86113631,0.33998104,-0.33998104])
    ome = np.array([0.34785485,0.34785485,0.65214515,0.65214515])
    xco = np.zeros(4)
    yco = np.zeros(4)
    ax = (x2-x1)/2
    bx = (x2+x1)/2
    ay = (y2-y1)/2
    by = (y2+y1)/2
    sl = math.sqrt(ax**2 + ay**2)
    eta1 = ay/sl
    eta2 = -ax/sl
    leta = math.sqrt(eta1**2 + eta2**2)
    b = 0
    a = 0
    c = 0
    d = 0
    for i in range(4):
        xco[i] = ax*csi[i]+bx
        yco[i] = ay*csi[i]+by
        ra = math.sqrt((xp-xco[i])**2 + (yp-yco[i])**2)
        rd1 = (xco[i] - xp)/ra
        rd2 = (yco[i] - yp)/ra
        rdn = rd1*eta1/leta + rd2*eta2/leta
        b = b - math.log(ra)*ome[i]*sl
        a = a + rdn*ome[i]*sl/ra
        d = d + (ra**2)*(math.log(ra) - 1)*ome[i]/sl
        c = c + rdn*ra*(2*math.log(ra)-1)*ome[i]*sl
    d = d/4
    c = c/(-4)
    return [a,b,c,d]

def Diag(x1,y1,x2,y2):
    ax = (x2-x1)/2
    ay = (y2-y1)/2
    sr = math.sqrt(ax**2 + ay**2)
    b = - sr *(math.log(sr/2) - 1)
    d = ((sr**3)/48)*(math.log(sr/2)-(4/3))
    return [b,d]

def PopulateGH():
    for i in range(NN):
        for j in range(NN):
            if i!=j:
                A[i,j] = OffDiag(xm[i],ym[i],x[j],y[j],x[j+1],y[j+1])[0]
                B[i,j] = OffDiag(xm[i],ym[i],x[j],y[j],x[j+1],y[j+1])[1]
                C[i,j] = OffDiag(xm[i],ym[i],x[j],y[j],x[j+1],y[j+1])[2]
                D[i,j] = OffDiag(xm[i],ym[i],x[j],y[j],x[j+1],y[j+1])[3]
            else:
                A[i,j] = - math.pi
                B[i,j] = Diag(x[j],y[j],x[j+1],y[j+1])[0]
                C[i,j] = 0
                D[i,j] = Diag(x[j],y[j],x[j+1],y[j+1])[1]
                
def FormSystem():
    system_freeterm[0:NN] = - np.dot(A,pD) - np.dot(B,pN)
    system_matrix[0:NN,0:NN] = C
    system_matrix[0:NN,NN:(2*NN)] = D
    system_matrix[NN:(2*NN),0:NN] = A
    system_matrix[NN:(2*NN),NN:(2*NN)] = B
    
def Values():
    for k in range(L):
        for j in range(NN):
            a = OffDiag(cx[k],cy[k],x[j],y[j],x[j+1],y[j+1])[0]
            b = OffDiag(cx[k],cy[k],x[j],y[j],x[j+1],y[j+1])[1]
            c = OffDiag(cx[k],cy[k],x[j],y[j],x[j+1],y[j+1])[2]
            d = OffDiag(cx[k],cy[k],x[j],y[j],x[j+1],y[j+1])[3]
            psi[k] = psi[k] + a*pD[j]+ b*pN[j] + c*wD[j] + d*wN[j]
        psi[k] = psi[k]/(2*math.pi)
    psi[0] = psi[0]*2
    psi[L-1] = psi[0]*2
                
PopulateGH()
FormSystem()
system_unknown = np.linalg.solve(system_matrix,system_freeterm)
wD = system_unknown[0:NN]
wN = system_unknown[NN:2*NN]
Values()

fig, ax = plt.subplots()
ax.plot(cy, psi)

ax.set(xlabel='vertical distance Y', ylabel='stream-function psi',
       title='Comparison at X-line')
ax.grid()

fig.savefig("test.png")
plt.show()
