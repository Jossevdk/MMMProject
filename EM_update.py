import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
import pandas as pd
import scipy.sparse as sp
import time
import psutil


#import QM_coupledback_Anton as QM
c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)

global g
g = True




def lu_full_pivot(A):
    n = A.shape[0]
    P = np.eye(n)  # Permutation matrix for row swaps
    Q = np.eye(n)  # Permutation matrix for column swaps
    U = A.copy()
    L = np.eye(n)
    
    for k in range(n):
        # Find the pivot index
        max_index = np.unravel_index(np.argmax(np.abs(U[k:, k:])), U[k:, k:].shape)
        pivot_row = max_index[0] + k
        pivot_col = max_index[1] + k
        
        # Swap rows in U and P
        U[[k, pivot_row], k:] = U[[pivot_row, k], k:]
        P[[k, pivot_row], :] = P[[pivot_row, k], :]
        
        # Swap columns in U and Q
        U[:, [k, pivot_col]] = U[:, [pivot_col, k]]
        Q[:, [k, pivot_col]] = Q[:, [pivot_col, k]]
        
        # Swap rows in L to maintain lower triangular form, only for columns before k
        if k > 0:
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Compute multipliers and eliminate below
        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
    
    return P, L, U, Q





### Source ###
class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma
        
        
    #This will call the function depending on which type of source you have    
    def J(self, t):
        #print(t)
        #return 10e7*np.cos(2*np.pi*2e7*t + 0.5)
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))#*np.cos(10*t/self.tc)



#### UCHIE ####
class UCHIE:
    def __init__(self, Nx, Ny, dx, dy, dt, QMxpos, pml_kmax = None, pml_nl = None):
        #dx = 0.005 # m
        #dy = 0.005
        #dt = 0.0000000001
        #Nx = 6
        #Ny = 6
        #pml_nl = 2
        #eps0 = 8.854 * 10**(-12)
        #mu0 = 4*np.pi * 10**(-7)
#
        #Z0 = np.sqrt(mu0/eps0)

        self.Nx = Nx
        self.Ny = Ny

        self.dx = dx
        self.dy = dy
        self.dt = c0*dt
        self.QMxpos = QMxpos



        self.X = np.zeros((5*Nx-1, Ny)) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
        A1 = np.zeros((Nx, Nx-1))
        np.fill_diagonal(A1, 1)
        np.fill_diagonal(A1[1:], 1)

        A2 = np.zeros((Nx, Nx+1))
        self.A2 = A2
        np.fill_diagonal(A2, 1)
        np.fill_diagonal(A2[:,1:], 1)

        D1 = np.zeros((Nx, Nx-1))
        np.fill_diagonal(D1, 1/dx)
        np.fill_diagonal(D1[1:], -1/dx)

        D2 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(D2, -1/dx)
        np.fill_diagonal(D2[:,1:], 1/dx)
        I_E = np.eye(Nx - 1)
        I_H = np.eye(Nx + 1)

        C = np.zeros((2*Nx, 2*Nx))
        np.fill_diagonal(C[:-1, 1:], 1)
        C[Nx*2-1, 0] = 1



        R = np.zeros((2 * Nx, 2 * Nx)) 
        R[0, Nx-1] = 1 
        for k in range(1, Nx):
            R[2*k-1 , k-1] = 1  
            R[2*k, Nx + k -1] = 1 
        R[2*Nx-1, 2*Nx-1] = 1   

    


        m = 10
        pml_kxmax = pml_kmax
        pml_sigmax_max = (m+1)/(150*np.pi*dx)
        
        pml_kx = np.array([1 + (pml_kxmax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        pml_sigmax = np.array([pml_sigmax_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        
        k_tot_E = np.hstack((pml_kx[::-1], np.ones(Nx-1 - 2*pml_nl), pml_kx))
        sigma_tot_E = np.hstack((pml_sigmax[::-1], np.zeros(Nx - 1 - 2*pml_nl), pml_sigmax))
        k_tot_H = np.hstack((pml_kx[::-1], np.ones(Nx+1 - 2*pml_nl), pml_kx))
        sigma_tot_H = np.hstack((pml_sigmax[::-1], np.zeros(Nx + 1 - 2*pml_nl), pml_sigmax))
        #print(k_tot_E, sigma_tot_E, k_tot_H, sigma_tot_H)
        # pml_kymax = 4
        # pml_sigmay_max = (m+1)/(150*np.pi*dy)
        # pml_ky = np.array([1 + (pml_kymax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        # pml_sigmay = np.array([pml_sigmay_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        #print(np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2))
       
        M11 = np.vstack((np.hstack((A1/self.dt,                np.zeros((Nx, Nx+1)))), \
                         np.hstack((np.zeros((Nx, Nx-1)), A2@np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)))))
        M12 = np.vstack((np.hstack((np.zeros((Nx, Nx-1)), D2, np.zeros((Nx, Nx-1)))), \
                         np.hstack((np.zeros((Nx, Nx-1)), np.zeros((Nx, Nx+1)), D1))))
        M21 = np.vstack((np.hstack((-I_E/self.dt,               np.zeros((Nx-1, Nx+1)))), \
                         np.hstack((np.zeros((Nx+1, Nx-1)),   -I_H/self.dt)), \
                         np.hstack((np.zeros((Nx-1, Nx-1)),    np.zeros((Nx-1, Nx+1))))))
        M22 = np.vstack((np.hstack((I_E/self.dt,               np.zeros((Nx-1, Nx+1)), np.zeros((Nx-1, Nx-1)))), \
                            np.hstack((np.zeros((Nx+1, Nx-1)),   I_H/self.dt, np.zeros((Nx+1, Nx-1)))), \
                            np.hstack((-I_E/self.dt,    np.zeros((Nx-1, Nx+1)), np.diag(k_tot_E/self.dt+Z0*sigma_tot_E/2)))))
        
        self.M22_inv = sp.csr_matrix(np.linalg.inv(M22))
        self.M12_M22inv = sp.csr_matrix(M12@self.M22_inv)
        self.M22inv_M21 = sp.csr_matrix(self.M22_inv@M21)
        #print(self.M22_inv.nnz)

        S = np.vstack((np.hstack((A1/self.dt,                D2,)), \
                       np.hstack((D1@np.linalg.inv(np.diag(k_tot_E/self.dt+Z0*sigma_tot_E/2))/self.dt, A2@np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)))))
       
        B = C@R@S@(R.T)

        P, L, U, Q = lu_full_pivot(B)
       

        self.RQULPCR = sp.csr_matrix((R.T)@Q@np.linalg.inv(U)@np.linalg.inv(L)@P@C@R)


        N1 = np.hstack((A1/self.dt,                np.zeros((Nx, Nx+1)),                          np.zeros((Nx, Nx-1)),    -D2,                       np.zeros((Nx, Nx-1))))
        N2 = np.hstack((np.zeros((Nx, Nx-1)),      A2@np.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2),  np.zeros((Nx, Nx-1)),     np.zeros((Nx, Nx+1)),     -D1))
        N3 = np.hstack((-I_E/self.dt,              np.zeros((Nx-1, Nx+1)),                        I_E/self.dt,              np.zeros((Nx-1, Nx+1)),   np.zeros((Nx-1, Nx-1))))
        N4 = np.hstack((np.zeros((Nx + 1, Nx-1)),  -I_H/self.dt,                                  np.zeros((Nx + 1, Nx-1)), I_H/self.dt,              np.zeros((Nx+1, Nx-1))))
        N5 = np.hstack((np.zeros((Nx - 1, Nx-1)),  np.zeros((Nx - 1, Nx+1)),                      -I_E/self.dt,             np.zeros((Nx - 1, Nx+1)), np.diag(k_tot_E/self.dt-Z0*sigma_tot_E/2)))
        
        self.N = sp.csr_matrix(np.vstack((N1, N2, N3, N4, N5)))    


        #explicit part
        self.ex2 = np.zeros((Nx+1, Ny+1))
        self.ex2old = np.zeros((Nx+1, Ny+1))
        self.ex1 = np.zeros((Nx+1, Ny+1))
        self.ex1old = np.zeros((Nx+1, Ny+1))
        self.ex0 = np.zeros((Nx+1, Ny+1))

        self.Y =   np.vstack((np.zeros((self.Nx, self.Ny)),self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy, np.zeros((self.Nx-1, self.Ny)), np.zeros((self.Nx+1, self.Ny)), np.zeros((self.Nx-1, self.Ny)) ))
         

        self.Betax_min = np.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2)
        self.Betay_min = np.eye(Nx+1)/self.dt
        self.Betaz_min = np.eye(Nx+1)/self.dt
        self.Betax_plus = np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)
        self.Betay_plus_inv = np.linalg.inv(np.eye(Nx+1)/self.dt)
        self.Betaz_plus_inv = np.linalg.inv(np.eye(Nx+1)/self.dt)

    def explicit(self):
        #this update takes ex at n to ex at n+1

        # self.ex2old = copy.deepcopy(self.ex2)
        # self.ex1old = copy.deepcopy(self.ex1)
        self.ex2old = self.ex2
        self.ex1old = self.ex1

        ex2 = self.ex2[:,1:-1] + self.dt/(self.dy)*(self.X[3*self.Nx-1:4*self.Nx,1:] - self.X[3*self.Nx-1:4*self.Nx,:-1])
        ex1 = self.Betay_plus_inv@(self.Betay_min@self.ex1[:,1:-1] + (ex2 - self.ex2old[:,1:-1])/self.dt)
        self.ex0[:,1:-1] = self.Betaz_plus_inv@(self.Betaz_min@self.ex0[:,1:-1] + self.Betax_plus@ex1 - self.Betax_min@self.ex1old[:,1:-1])
        
        self.ex2[:,1:-1] = ex2
        self.ex1[:,1:-1] = ex1
        
  

    def implicit(self, n, source, JQM):
        self.Y[self.Nx:2*self.Nx , :] = self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy
        self.Y[self.Nx + int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)


        #print(self.Y[int(self.QMxpos/self.dx), :].shape)
        #print(JQM.shape)
        self.Y[ int(self.QMxpos), :] += JQM[:]



        rhs = self.N@self.X + self.Y
        v1 = rhs[:2*self.Nx, :]
        v2 = rhs[2*self.Nx:, :]
        p = v1 - self.M12_M22inv@v2
        self.X[:2*self.Nx, :] = self.RQULPCR@p
        self.X[2*self.Nx:, :] = self.M22_inv@v2 - self.M22inv_M21@self.X[:2*self.Nx, :]

    def Update(self,n, source, JQM):
        self.implicit(n, source, JQM)
        self.explicit()
        return Z0*self.X[4*self.Nx:5*self.Nx-1,:]

    # def calculate(self, Nt, source):
    #     data_time = []
    #     data = []

    #     for n in range(0, Nt):
    #         self.implicit(n, source)
    #         self.explicit()
    #         data_time.append(self.dt*n)
    #         #data.append(copy.deepcopy((Z0*self.ex0.T)))
    #         #data.append((Z0*self.ex0.T))
    #         data.append(copy.deepcopy((self.X[3*self.Nx-1:4*self.Nx,:].T)))
            
        
    #     return data_time, data
    # def animate_field(self, t, data):
    #     fig, ax = plt.subplots()

    #     ax.set_xlabel("x-axis [k]")
    #     ax.set_ylabel("y-axis [l]")
    #     # ax.set_xlim(0, Nx*dx)
    #     # ax.set_ylim(0, Ny*dy)

    #     label = "Field"
        
    #     # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source
    #     cax = ax.imshow(data[0],vmin = -1e-16, vmax = 1e-16)
    #     ax.set_title("T = 0")

    #     def animate_frame(i):
    #         cax.set_array(data[i])
    #         ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
    #         return cax

    #     global anim
        
    #     anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
    #     plt.show()




# dx = 1e-10 # m
# dy = 0.125e-9# ms

# Sy = 0.8 # !Courant number, for stability this should be smaller than 1
# dt = Sy*dy/c0
# #print(dt)
# Nx = 400
# Ny = 400
# Nt = 400

# pml_nl = 20
# pml_kmax = 4
# eps0 = 8.854 * 10**(-12)
# mu0 = 4*np.pi * 10**(-7)
# Z0 = np.sqrt(mu0/eps0)


# xs = Nx*dx/2 
# ys = Ny*dy/2

# tc = dt*Nt/4
# #print(tc)
# sigma = tc/10
# source = Source(xs, ys, 1, tc, sigma)


# scheme = UCHIE(Nx, Ny, dx, dy, dt, pml_kmax = pml_kmax, pml_nl = pml_nl)
# start_time = time.time()

# data_time, data = scheme.calculate(Nt, source)

# process = psutil.Process()
# print("Memory usage:", process.memory_info().rss) # print memory usage
# print("CPU usage:", process.cpu_percent()) # print CPU usage

# end_time = time.time()


# print("Execution time: ", end_time - start_time, "seconds")

# scheme.animate_field(data_time, data)
         