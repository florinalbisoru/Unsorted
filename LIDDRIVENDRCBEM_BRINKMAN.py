# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:36:32 2019

@author: Florin Albisoru

Description : DRBEM lid - driven cavity - Brinkman system
"""
import math
import numpy as np
import matplotlib.pyplot as plt

Da = 0.25
phi = 0.2
Lambda = 1

a = phi/(Da*Lambda)

N = 128
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
x_vals = np.linspace(0.01,0.99,int(math.sqrt(L)))
y_vals = np.linspace(0.01,0.99,int(math.sqrt(L)))
X, Y = np.meshgrid(x_vals,y_vals)
for i in range(int(math.sqrt(L))):
    xnode[N+i*int(math.sqrt(L)):N+(i+1)*int(math.sqrt(L))] = X[i,:]
    ynode[N+i*int(math.sqrt(L)):N+(i+1)*int(math.sqrt(L))] = Y[i,:]
    
#conditiile pe frontiera
pd = np.zeros(N+L)
pn = np.zeros(N+L)
wd = np.zeros(N+L)
wn = np.zeros(N+L)

pn[int(N/2):int(3*N/4)] = -1*np.ones(int(N/4))

F = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        F[k,j] = 1 + r
        
Finv = np.linalg.inv(F)
        
#cuadraturi Gauss
csi = np.array([0.339981,-0.339981,0.861136,-0.861136])
wg = np.array([0.652145,0.652145,0.347855,0.347855])

#constructia matricilor
G = np.zeros((N+L,N+L))
H = np.zeros((N+L,N+L))

for i in range(N+L):
    for j in range(N):
        lj = math.sqrt((xedge[j+1]-xedge[j])**2 + (yedge[j+1]-yedge[j])**2)
        if i != j :
            g = 0
            h = 0
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
            G[i,j] = lj*g/(4*math.pi)
            H[i,j] = -lj*h/(4*math.pi)
        else:
            G[i,j] = lj*(math.log(2/lj) + 1)/(2*math.pi)
            H[i,j] = 1/2
            
H[N:N+L,N:N+L] = np.eye(L)

uh = np.zeros((N+L,N+L))
qh = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        uh[k,j] = (r**2)/4 + (r**3)/9
        
for k in range(N):
    lk = math.sqrt((xedge[k+1]-xedge[k])**2 + (yedge[k+1]-yedge[k])**2)
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        difx = xedge[k+1]-xedge[k]
        dify = yedge[k+1]-yedge[k]
        eta1 = dify/lk
        eta2 = -difx/lk
        qh[k,j] = (0.5+r/3)*((xnode[k]-xnode[j])*eta1 + (ynode[k]-ynode[j])*eta2)
        
S = np.zeros((N+L,N+L))         
S = S + np.matmul(G,qh) - np.matmul(H,uh)
S = np.matmul(S,Finv)

system_freeterm = np.zeros(2*N)
system_matrix = np.zeros((2*N,2*N))

system_freeterm[0:N] = -np.matmul(H[0:N,0:N],pd[0:N]) + np.matmul(G[0:N,0:N],pn[0:N])
system_matrix[0:N,0:N] = S[0:N,0:N]
system_matrix[N:2*N,0:N] = H[0:N,0:N] - a*S[0:N,0:N]
system_matrix[N:2*N,N:2*N] = -G[0:N,0:N]

system_solution = np.linalg.solve(system_matrix,system_freeterm)
wd[0:N] = system_solution[0:N]
wn[0:N] = system_solution[N:2*N]

rezpsi = np.zeros(L)
for i in range(L):
    rezpsi[i] = -np.dot(H[N+i,0:N],pd[0:N]) + np.dot(G[N+i,0:N],pn[0:N]) - np.matmul(S[N+i,0:N],wd[0:N])
    
temp = int(math.sqrt(L)) 
   
Z = np.zeros((int(math.sqrt(L)),int(math.sqrt(L))))
for i in range(int(math.sqrt(L))):
    Z[i,:] = rezpsi[int(math.sqrt(L))*i:int(math.sqrt(L))*(i+1)]    
    
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)

ax.set_title('Contour Plot')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()