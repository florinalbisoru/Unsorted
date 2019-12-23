# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:45:57 2019

@author: Florin Albisoru

Nota : am dubii destul de mari ca la implementarea acestuia s-ar putea sa fi
gresit
"""

import math
import numpy as np
import matplotlib.pyplot as plt

N = 400
L = 400

xedge = np.zeros(N+1)
yedge = np.zeros(N+1)

xedge[0:int(N/4)] = np.linspace(0,1,num = int(N/4),endpoint = False)
xedge[int(N/4):int(N/2)] = np.ones(int(N/4))
xedge[int(N/2):int(3*N/4)] = np.linspace(1,0,num = int(N/4),endpoint = False)

yedge[int(N/4):int(N/2)] = np.linspace(0,1,num = int(N/4),endpoint = False)
yedge[int(N/2):int(3*N/4)] = np.ones(int(N/4))
yedge[int(3*N/4):N] = np.linspace(1,0,num = int(N/4),endpoint = False)

xnode = np.zeros(N+L)
ynode = np.zeros(N+L)
xnode[0:N] = (xedge[0:N] + xedge[1:N+1])/2
ynode[0:N] = (yedge[0:N] + yedge[1:N+1])/2
#xnode[N:N+L] = 0.1*np.ones(L)
#ynode[N:N+L] = np.linspace(0,1,num = L)

x_vals = np.linspace(0.01,0.99,int(L/20))
y_vals = np.linspace(0.01,0.99,int(L/20))
X, Y = np.meshgrid(x_vals,y_vals)
for i in range(int(L/20)):
    xnode[N+i*int(L/20):N+(i+1)*int(L/20)] = X[i,:]
    ynode[N+i*int(L/20):N+(i+1)*int(L/20)] = Y[i,:]

#conditiile pe frontiera
pd = np.zeros(N)
pn = np.zeros(N)
wd = np.zeros(N)
wn = np.zeros(N)

pn[int(N/2):int(3*N/4)] = -1*np.ones(int(N/4))

#cuadraturi Gauss
csi = np.array([0.339981044,-0.339981044,0.861136312,-0.861136312])
wg = np.array([0.652145155,0.652145155,0.347854845,0.347854845])

G = np.zeros((N+L,N))
H = np.zeros((N+L,N))
A = np.zeros((N+L,N))
B = np.zeros((N+L,N))

for i in range(N+L):
    for j in range(N):
        lj = math.sqrt((xedge[j+1]-xedge[j])**2 + (yedge[j+1]-yedge[j])**2)
        if i != j :
            g = 0
            h = 0
            a = 0
            b = 0
            sumx = xedge[j+1]+xedge[j]
            sumy = yedge[j+1]+yedge[j]
            difx = xedge[j+1]-xedge[j]
            dify = yedge[j+1]-yedge[j]
            eta1 = dify/lj
            eta2 = -difx/lj
            for k in range(4):
                xcsi = difx/2*csi[k] + sumx/2
                ycsi = dify/2*csi[k] + sumy/2
                r = math.sqrt((xcsi - xnode[i])**2 + (ycsi - ynode[i])**2)
                g = g + math.log(1/r)*wg[k]
                h = h + (1/(r**2))*wg[k]*(eta1*(xcsi - xnode[i]) + eta2*(ycsi-ynode[i]))
                a = a + r*wg[k]*(2*math.log(1/r) + 1)*(eta1*(xcsi - xnode[i]) + eta2*(ycsi-ynode[i]))
                b = b + (r**2)*wg[k]*(math.log(1/r)+1)
            G[i,j] = lj*g/(4*math.pi)
            H[i,j] = -lj*h/(4*math.pi)
            A[i,j] = lj*a/(16*math.pi)
            B[i,j] = lj*b/(16*math.pi)
        else:
            G[i,j] = lj*(math.log(2/lj) + 1)/(2*math.pi)
            H[i,j] = 1/2
            A[i,j] = 0
            B[i,j] = (lj**3)/(96*math.pi)*((4/3) - math.log(lj/2))
            
system_freeterm = np.zeros(2*N)
system_matrix = np.zeros((2*N,2*N))

system_freeterm[N:2*N] = - np.matmul(H[0:N,0:N],pd) + np.matmul(G[0:N,0:N],pn)
system_matrix[0:N,0:N] = H[0:N,0:N]
system_matrix[0:N,N:2*N] = - G[0:N,0:N]
system_matrix[N:2*N,0:N] = - A[0:N,0:N]
system_matrix[N:2*N,N:2*N] = B[0:N,0:N]

system_sol = np.linalg.solve(system_matrix,system_freeterm)
wd = system_sol[0:N] 
wn = system_sol[N:2*N]

rezpsi = np.zeros(L)
for i in range(L):
    rezpsi[i] = -np.dot(H[N+i,:],pd) + np.dot(G[N+i,:],pn) + np.dot(A[N+i,:],wd) - np.dot(B[N+i,:],wn)

"""
fig= plt.figure()
axes= fig.add_axes([0.1,0.1,0.8,0.8])  
axes.plot(ynode[N:N+L],rezpsi)
plt.show()    
"""

rezw = np.zeros(L)
for i in range(L):
    rezw[i] = -np.dot(H[N+i,:],wd) + np.dot(G[N+i,:],wn)
    
U = np.zeros((int(L/20),int(L/20)))
for i in range(int(L/20)):
    U[i,:] = rezw[int(L/20)*i:int(L/20)*(i+1)]

Z = np.zeros((int(L/20),int(L/20)))
for i in range(int(L/20)):
    Z[i,:] = rezpsi[int(L/20)*i:int(L/20)*(i+1)]

"""
#prelucram aici valorile de pe frontiera pt ca am ales gridul sa contina si
#pct de frontiera => tb sa impartim la 1/2 ca sa fie corecte valorile
Z[:,0] = Z[:,0]/2
Z[:,int(L/20)-1] = Z[:,int(L/20)-1]/2
Z[0,1:int(L/20)-1] = Z[0,1:int(L/20)-1]/2
Z[int(L/20)-1,1:int(L/20)-1] = Z[int(L/20)-1,1:int(L/20)-1]/2

U[:,0] = U[:,0]/2
U[:,int(L/20)-1] = U[:,int(L/20)-1]/2
U[0,1:int(L/20)-1] = U[0,1:int(L/20)-1]/2
U[int(L/20)-1,1:int(L/20)-1] = U[int(L/20)-1,1:int(L/20)-1]/2
"""
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)

ax.set_title('Contour Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()

fig2 = plt.figurefig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax2 = fig2.add_axes([left, bottom, width, height]) 

cp = plt.contourf(X, Y, U)
plt.colorbar(cp)

ax2.set_title('Contour Plot')
ax2.set_xlabel('x (cm)')
ax2.set_ylabel('y (cm)')
plt.show()
