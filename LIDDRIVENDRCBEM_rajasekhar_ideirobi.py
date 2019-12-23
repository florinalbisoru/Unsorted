# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:39:33 2019

@author: Florin Albisoru

Description : preluam matricile H si G, gasim F,uh,qh si rezolvam problema
abordata in articolul lui Raja Sekhar cu idei de la Robi si din BPW

Nota: N=400,L=400 cu linspace de la 0.01 la 0.99 functioneaza bini di tat
Nota2 : e cel mai apropiat de ceea ce trebuie la lid-driven
"""
import numpy as np
import math
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

x_vals = np.linspace(0,1,int(math.sqrt(L)))
y_vals = np.linspace(0,1,int(math.sqrt(L)))
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

#cuadraturi Gauss
csi = np.array([0.339981044,-0.339981044,0.861136312,-0.861136312])
wg = np.array([0.652145155,0.652145155,0.347854845,0.347854845])

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

F = np.zeros((N+L,N+L))

for k in range(N+L):
    for j in range(N+L):
        r = math.sqrt((xnode[k]-xnode[j])**2 + (ynode[k]-ynode[j])**2)
        F[k,j] = 1+r
        
Finv = np.linalg.inv(F)

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
        
S = np.matmul(H,uh) - np.matmul(G,qh)
S = np.matmul(S,Finv)

system_matrix = np.zeros((2*(N+L),2*(N+L)))
system_freeterm = np.zeros(2*N+2*L)

#coloana1
system_matrix[N:N+L,0:L] = np.eye(L)

#coloana2
system_matrix[0:N,L:L+N] = S[0:N,0:N]
system_matrix[N:N+L,L:L+N] = S[N:N+L,0:N]
system_matrix[N+L:N+L+N,L:L+N] = H[0:N,0:N]
system_matrix[2*N+L:2*N+2*L,L:L+N] = H[N:N+L,0:N]

#coloana3
system_matrix[0:N,L+N:N+2*L] = S[0:N,N:N+L]
system_matrix[N:N+L,L+N:N+2*L] = S[N:N+L,N:N+L]
system_matrix[2*N+L:2*N+2*L,L+N:N+2*L] = np.eye(L)

#coloana4
system_matrix[N+L:2*N+L,2*L+N:2*L+2*N] = - G[0:N,0:N]
system_matrix[2*N+L:2*N+2*L,2*L+N:2*L+2*N] = - G[N:N+L,0:N]

system_freeterm[0:N] = np.matmul(G[0:N,0:N],pn[0:N])
system_freeterm[N:N+L] = np.matmul(G[N:N+L,0:N],pn[0:N]) 

system_solution = np.linalg.solve(system_matrix,system_freeterm)
pd[N:N+L] = system_solution[0:L]
wd[N:N+L] = system_solution[N+L:N+2*L]

Z = np.zeros((int(math.sqrt(L)),int(math.sqrt(L))))
for i in range(int(math.sqrt(L))):
    Z[i,:] = pd[N+int(math.sqrt(L))*i:N+int(math.sqrt(L))*(i+1)]
#verifica sa iti dea bine si in cazul in care consideri de la 0 la 1 cu frontiera
Z[0,:] = np.zeros(int(math.sqrt(L)))
Z[:,0] = np.zeros(int(math.sqrt(L)))
Z[int(math.sqrt(L))-1,:] = np.zeros(int(math.sqrt(L)))
Z[:,int(math.sqrt(L))-1] = np.zeros(int(math.sqrt(L)))
  
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)

ax.set_title('Streamline - function')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

U = np.zeros((int(math.sqrt(L)),int(math.sqrt(L))))
for i in range(int(math.sqrt(L))):
    U[i,:] = wd[N+int(math.sqrt(L))*i:N+int(math.sqrt(L))*(i+1)]
    
#analog cu vorticitatea
U[0,:] = U[0,:]/2
U[int(math.sqrt(L))-1,:] = U[int(math.sqrt(L))-1,:]/2
U[1:int(math.sqrt(L))-1,0] = U[1:int(math.sqrt(L))-1,0]/2
U[1:int(math.sqrt(L))-1,int(math.sqrt(L))-1] = U[1:int(math.sqrt(L))-1,int(math.sqrt(L))-1]/2
    
fig2 = plt.figurefig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax2 = fig2.add_axes([left, bottom, width, height]) 

cp = plt.contourf(X, Y, U)
plt.colorbar(cp)

ax2.set_title('Vorticity')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.show()


