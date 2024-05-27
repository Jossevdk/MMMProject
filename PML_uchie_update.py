import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
import pandas as pd
from scipy.sparse import csr_matrix
import torch as th
import time
import psutil
import time
import psutil

from PMLX import PML_X
from PMLY import PML_Y


c0 = 299792458  # Speed of light in vacuum
eps0 = 8.854 * 10**(-12)
mu0 = 4 * np.pi * 10**(-7)

Z0 = np.sqrt(mu0 / eps0)

### Source ###
class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma

    def J(self, t):
        return self.J0 * np.exp(-(t - self.tc)**2.0 / (2 * self.sigma**2.0))

### UCHIE ###
class UCHIE:
    def __init__(self, Nx, Ny, NyQM,dx, dy, dt, QMxpos, pml_kmax=None, pml_nl=None, m=None):
        self.Nx = Nx
        self.Ny = Ny
        self.NyQM = NyQM
        self.dx = dx
        self.dy = dy
        self.dt = c0 * dt
        self.X = np.zeros((2 * Nx+2, Ny))
        self.Y = np.zeros((2 * Nx+2, Ny))
        self.QMxpos = QMxpos

        self.data_E = []

        A1 = np.zeros((Nx+2, Nx+1))
        np.fill_diagonal(A1, 1)
        np.fill_diagonal(A1[1:], 1)

        A2 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(A2, 1)
        np.fill_diagonal(A2[:,1:], 1)
        self.A2 = A2

        D1 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(D1, -1/dx)
        np.fill_diagonal(D1[:,1:], 1/dx)

        D2 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(D2, -1/dx)
        np.fill_diagonal(D2[:,1:], 1/dx)
        D2 = np.vstack((np.zeros(Nx+1), D2, np.zeros(Nx+1)))

    

        M_midden1 = np.hstack((A1 / self.dt, D2))
        M_midden2 = np.hstack((D1, A2 / self.dt))
        M = np.vstack((M_midden1, M_midden2))

        N_midden1 = np.hstack((A1 / self.dt, -D2))
        N_midden2 = np.hstack((-D1, A2 / self.dt))
        N = np.vstack((N_midden1, N_midden2))
        self.M_inv = np.linalg.inv(M)
        self.M_N = self.M_inv @ N
    
        self.ex0 = np.zeros((Nx+1, Ny + 1))
        if pml_kmax is not None and pml_nl is not None and m is not None:
            self.use_pml = True
            self.PMLR = PML_X(pml_nl + 2, Ny, dx, dy, dt, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
            self.PMLL = PML_X(pml_nl + 2, Ny, dx, dy, dt, reverse= True,  pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
            self.PMLU = PML_Y(Nx, pml_nl+2, dx, dy, dt, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
            self.PMLD = PML_Y(Nx, pml_nl+2, dx, dy, dt, reverse=True, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
    def explicit(self):
        self.ex0[:, 1:-1] += self.dt / (self.dy) * (self.X[self.Nx+1:2 * self.Nx+2, 1:] - self.X[self.Nx+1:2 * self.Nx+2, :-1])

    def implicit(self, n, source, JQM):
        slice = int(1/2*(self.Ny-self.NyQM))
        #self.Y[self.QMxpos, slice :-slice]+= -2 * (1 / Z0) * JQM
        self.Y[self.Nx+2:2*self.Nx+2 , :] = self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy
        self.Y[self.Nx+2 + int(source.x / self.dx), int(source.y / self.dy)] += -2 * (1 / Z0) * source.J(n * self.dt / c0)/self.dx/self.dy
        

        self.X = self.M_N @ self.X + self.M_inv @ self.Y

    def update(self, n, source, JQM):

        
        
        self.implicit(n, source, JQM)
        self.explicit()
        if self.use_pml:
            self.PMLL.X[4*self.PMLL.Nx+3, :] = self.X[self.Nx+2, :] #Hz CORRECT
            self.PMLL.X[self.PMLL.Nx, :] = self.X[1, :] #Ey CORRECT
            self.PMLL.ex0[-1, :] = self.ex0[1, :] #Ex CORRECT
            self.PMLL.update(n)
            self.X[self.Nx+1, :] = self.PMLL.X[4*self.PMLL.Nx+2, :] #Hz CORRECT
            self.X[0, :] = self.PMLL.X[self.PMLL.Nx-1, :] #Ey CORRECT
            self.ex0[0, :] = self.PMLL.ex0[-2, :] #Ex CORRECT

            

            self.PMLR.X[3*self.PMLR.Nx+3, :] = self.X[-2, :]
            self.PMLR.X[0, :] = self.X[-self.Nx-3, :]
            self.PMLR.ex0[0, :] = self.ex0[-2, :]
            self.PMLR.update(n)
            self.X[-1, :] = self.PMLR.X[3*self.PMLR.Nx+4, :]
            self.X[-self.Nx-2, :] = self.PMLR.X[1, :]
            self.ex0[-1, :] = self.PMLR.ex0[1, :]

            self.PMLU.X[2*self.PMLU.Nx+2:3*self.PMLU.Nx+3, 1] = self.X[self.Nx+1:2*self.Nx+2, -1] #Hz CORRECT
            self.PMLU.X[0:self.PMLU.Nx+1, 1] = self.X[0:self.Nx+1, -1] #Ey CORRECT
            self.PMLU.ex0[:, 1] = self.ex0[:, -1] #Ex CORRECT
            self.PMLU.update(n)
            self.X[self.Nx+1:2*self.Nx+2, -2]= self.PMLU.X[2*self.PMLU.Nx+2:3*self.PMLU.Nx+3, 0] #Hz CORRECT
            self.X[0:self.Nx+1, -2] = self.PMLU.X[0:self.PMLU.Nx+1, 0] #Ey CORRECT
            self.ex0[:, -2] = self.PMLU.ex0[:, 0] #Ex CORRECT

            self.PMLD.X[2*self.PMLD.Nx+2:3*self.PMLD.Nx+3, -2] = self.X[self.Nx+1:2*self.Nx+2, 0] #Hz CORRECT
            self.PMLD.X[0:self.PMLD.Nx+1, -2] = self.X[0:self.Nx+1, 0] #Ey CORRECT
            self.PMLD.ex0[:, -2] = self.ex0[:, 0] #Ex CORRECT
            self.PMLD.update(n)
            self.X[self.Nx+1:2*self.Nx+2, 1]= self.PMLD.X[2*self.PMLD.Nx+2:3*self.PMLD.Nx+3, -1] #Hz CORRECT
            self.X[0:self.Nx+1, 1] = self.PMLD.X[0:self.PMLD.Nx+1, -1] #Ey CORRECT
            self.ex0[:, 1] = self.PMLD.ex0[:, -1] #Ex CORRECT
        self.data_E.append(Z0 * self.X[:self.Nx - 1, :].T)
        return Z0 * self.X[:self.Nx - 1, :].T

    # def calculate(self, Nt, source):
    #     data_time = []
    #     data = []
    #     tracker =[[], []]
    #     for n in range(Nt):
    #         self.update(n, source)
    #         if n % 50 == 0:
    #             data_time.append(self.dt * n)
    #             data.append(Z0*self.ex0.T.copy())
    #             tracker[0].append((Z0*self.ex0[self.Nx//4,self.Ny//2]).copy())
    #             tracker[1].append((Z0*self.ex0[3*self.Nx//4,self.Ny//2]).copy())
            

    #     return data_time, data, tracker
    def animate_field(self):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source
        self.data_E = self.data_E[::50]
        cax = ax.imshow(self.data_E[3],vmin = -2e-13, vmax = 2e-13)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(self.data_E[i])
            #ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(self.data_E)), interval=2)
        plt.show()

# dx = 1e-10 # m
# dy = 0.125e-9# ms

# Sy = 0.8 # !Courant number, for stability this should be smaller than 1
# dt = Sy*dy/c0
# #print(dt)
# Nx = 300
# Ny = 300
# Nt = 1000

# pml_nl = 10
# pml_kmax = 4
# m = 10

# eps0 = 8.854 * 10**(-12)
# mu0 = 4*np.pi * 10**(-7)
# Z0 = np.sqrt(mu0/eps0)


# xs = Nx*dx/2 
# ys = Ny*dy/2

# tc = dt*Nt/4
# #print(tc)
# sigma = tc/10
# source = Source(xs, ys, 1, tc, sigma)

# scheme = UCHIE(Nx, Ny, dx, dy, dt, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
# start_time = time.time()
# data_time, data, tracker = scheme.calculate(Nt, source)

# plt.plot(data_time, tracker[0])
# plt.plot(data_time, tracker[1])
# end_time = time.time()
# print("Execution time:", end_time - start_time, "seconds")

# # Optionally animate or plot data here
# scheme.animate_field(data_time, data)
