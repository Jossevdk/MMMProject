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


c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)


device = th.device("cuda" if th.cuda.is_available() else "cpu")

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
        #print(t, self.tc, self.sigma)
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))



#### UCHIE ####
class UCHIE:
    def __init__(self, Nx, Ny, dx, dy, dt, pml_kmax = None, pml_nl = None):
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
        self.X = th.zeros((5*Nx-1, Ny)).to(device) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
        self.Y = th.zeros((5*Nx-1, Ny)).to(device)

        A1 = th.zeros((Nx, Nx-1))
        A1[th.arange(Nx-1), th.arange(Nx-1)] = 1
        A1[th.arange(1, Nx), th.arange(Nx-1)] = 1  # Adjusted indices

        A2 = th.zeros((Nx, Nx+1))
        A2[th.arange(Nx), th.arange(Nx)] = 1
        A2[th.arange(Nx), th.arange(1, Nx+1)] = 1
        
        self.A2 = A2.to(device)

        D1 = th.zeros((Nx, Nx-1))
        D1[th.arange(Nx-1), th.arange(Nx-1)] = 1/dx
        D1[th.arange(1, Nx), th.arange(Nx-1)] = -1/dx

        D2 = th.zeros((Nx, Nx+1))
        D2[th.arange(Nx), th.arange(Nx)] = -1/dx
        D2[th.arange(Nx), th.arange(1, Nx+1)] = 1/dx


       
        I_E = th.eye(Nx - 1)
        I_H = th.eye(Nx + 1)
        m = 10
        pml_kxmax = pml_kmax
        pml_sigmax_max = (m+1)/(150*np.pi*dx)
        
        pml_kx = th.tensor([1 + (pml_kxmax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        pml_sigmax = th.tensor([pml_sigmax_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        
        k_tot_E = th.hstack((pml_kx.flip(dims=[0]), th.ones(Nx-1 - 2*pml_nl), pml_kx))
        sigma_tot_E = th.hstack((pml_sigmax.flip(dims=[0]), th.zeros(Nx - 1 - 2*pml_nl), pml_sigmax))
        k_tot_H = th.hstack((pml_kx.flip(dims=[0]), th.ones(Nx+1 - 2*pml_nl), pml_kx))
        sigma_tot_H = th.hstack((pml_sigmax.flip(dims=[0]), th.zeros(Nx + 1 - 2*pml_nl), pml_sigmax))
        print(k_tot_E, sigma_tot_E, k_tot_H, sigma_tot_H)
        # pml_kymax = 4
        # pml_sigmay_max = (m+1)/(150*np.pi*dy)
        # pml_ky = np.array([1 + (pml_kymax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        # pml_sigmay = np.array([pml_sigmay_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        #print(np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2))
        M1 = th.hstack((A1/self.dt,                th.zeros((Nx, Nx+1)),                          th.zeros((Nx, Nx-1)),     D2,                       th.zeros((Nx, Nx-1))))
        M2 = th.hstack((th.zeros((Nx, Nx-1)),      th.mm(A2,th.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)),  th.zeros((Nx, Nx-1)),     th.zeros((Nx, Nx+1)),     D1))
        M3 = th.hstack((-I_E/self.dt,              th.zeros((Nx-1, Nx+1)),                        I_E/self.dt,              th.zeros((Nx-1, Nx+1)),   th.zeros((Nx-1, Nx-1))))
        M4 = th.hstack((th.zeros((Nx + 1, Nx-1)),  -I_H/self.dt,                                  th.zeros((Nx + 1, Nx-1)), I_H/self.dt,              th.zeros((Nx+1, Nx-1))))
        M5 = th.hstack((th.zeros((Nx - 1, Nx-1)),  th.zeros((Nx - 1, Nx+1)),                      -I_E/self.dt,             th.zeros((Nx - 1, Nx+1)), th.diag(k_tot_E/self.dt+Z0*sigma_tot_E/2)))
        
        N1 = th.hstack((A1/self.dt,                th.zeros((Nx, Nx+1)),                          th.zeros((Nx, Nx-1)),    -D2,                       th.zeros((Nx, Nx-1))))
        N2 = th.hstack((th.zeros((Nx, Nx-1)),      th.mm(A2,th.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2)),  th.zeros((Nx, Nx-1)),     th.zeros((Nx, Nx+1)),     -D1))
        N3 = th.hstack((-I_E/self.dt,              th.zeros((Nx-1, Nx+1)),                        I_E/self.dt,              th.zeros((Nx-1, Nx+1)),   th.zeros((Nx-1, Nx-1))))
        N4 = th.hstack((th.zeros((Nx + 1, Nx-1)),  -I_H/self.dt,                                  th.zeros((Nx + 1, Nx-1)), I_H/self.dt,              th.zeros((Nx+1, Nx-1))))
        N5 = th.hstack((th.zeros((Nx - 1, Nx-1)),  th.zeros((Nx - 1, Nx+1)),                      -I_E/self.dt,             th.zeros((Nx - 1, Nx+1)), th.diag(k_tot_E/self.dt-Z0*sigma_tot_E/2)))
        
        M = th.vstack((M1, M2, M3, M4, M5))
        
        self.M_inv = th.linalg.inv(M).to(device)
        N = th.vstack((N1, N2, N3, N4, N5)).to(device)
        self.M_N = th.mm(self.M_inv,N).to(device)

     


        #explicit part
        self.ex2 = th.zeros((Nx+1, Ny+1)).to(device)
        self.ex2old = th.zeros((Nx+1, Ny+1)).to(device)
        self.ex1 = th.zeros((Nx+1, Ny+1)).to(device)
        self.ex1old = th.zeros((Nx+1, Ny+1)).to(device)
        self.ex0 = th.zeros((Nx+1, Ny+1)).to(device)


        

        self.Betax_min = th.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2).to(device)
        self.Betay_min = (th.eye(Nx+1)/self.dt).to(device)
        self.Betaz_min = (th.eye(Nx+1)/self.dt).to(device)   
        self.Betax_plus = th.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2).to(device)
        self.Betay_plus_inv = th.inverse(th.eye(Nx+1)/self.dt).to(device)
        self.Betaz_plus_inv = th.inverse(th.eye(Nx+1)/self.dt).to(device)
        
        self.Betay = th.mm(self.Betay_plus_inv, self.Betay_min)
        self.Betaz = th.mm(self.Betaz_plus_inv, self.Betaz_min)
        self.Betazxplus = th.mm(self.Betaz_plus_inv, self.Betax_plus)
        self.Betazxmin = th.mm(self.Betaz_plus_inv, self.Betax_min)

    def explicit(self):
        self.ex2old = copy.deepcopy(self.ex2)
        self.ex1old = copy.deepcopy(self.ex1)

        self.ex2[:,1:-1] = self.ex2[:,1:-1] + self.dt/(self.dy)*(self.X[3*self.Nx-1:4*self.Nx,1:] - self.X[3*self.Nx-1:4*self.Nx,:-1])
        self.ex1[:,1:-1] = th.mm(self.Betay_plus_inv, th.mm(self.Betay_min, self.ex1[:,1:-1]) + (self.ex2[:,1:-1] - self.ex2old[:,1:-1])/self.dt)
        self.ex0[:,1:-1] = th.mm(self.Betaz_plus_inv, th.mm(self.Betaz_min, self.ex0[:,1:-1]) + th.mm(self.Betax_plus, self.ex1[:,1:-1]) - th.mm(self.Betax_min, self.ex1old[:,1:-1]))
        
  

    def implicit(self, n, source):
        #S_ = th.zeros((self.Nx, self.Ny))
        #S_[int(source.x/self.dx), int(source.y/self.dy)] = -2*(1/Z0)*source.J(n*self.dt/c0)
        #Y = th.vstack((th.zeros((self.Nx, self.Ny)),S_ + th.mm(self.A2, (self.ex0[:, 1:] - self.ex0[:, :-1]))/self.dy, th.zeros((self.Nx-1, self.Ny)), th.zeros((self.Nx+1, self.Ny)), th.zeros((self.Nx-1, self.Ny)) ))
        self.Y[self.Nx:2*self.Nx , :] =  th.mm(self.A2, (self.ex0[:, 1:] - self.ex0[:, :-1]))/self.dy
        self.Y[self.Nx + int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)
        #S = np.zeros((5*self.Nx-1, self.Ny))
        #S[self.Nx-1 + int(source.x/self.dx), int(source.y/self.dy)] = -2*(1/Z0)*source.J(n*self.dt/c0)*self.dt/c0
        #self.X[self.Nx-1 + int(source.x/self.dx), int(source.y/self.dy)] += -2000*(1/Z0)*source.J(n*self.dt/c0)*self.dt/c0
        
        self.X = th.mm(self.M_N, self.X )+ th.mm(self.M_inv, self.Y)
        #self.X[self.Nx-1 + int(source.x/self.dx), int(source.y/self.dy)] +=   -2000000000000000*(1/Z0)*source.J(n*self.dt/c0)
        #self.X[self.Nx-1 + int(source.x/self.dx), 1+int(source.y/self.dy)] += -2000000000000000*(1/Z0)*source.J(n*self.dt/c0)
        
        #print(self.X[self.Nx-1 + int(source.x/self.dx), int(source.y/self.dy)])
        # print(np.shape(self.X))

    def Update(self,n, source):
        self.implicit(n, source)
        self.explicit()
        return Z0*self.X[4*self.Nx:5*self.Nx-1,:].to("cpu").numpy()

    def calculate(self, Nt, source):
        data_time = []
        data = []
        tracker = []

        for n in range(0, Nt):
            self.implicit(n, source)
            self.explicit()
            if n % 5 == 0:
                data_time.append(self.dt*n)
                tracker.append(copy.deepcopy((self.X[3*self.Nx-1 +self.Nx//3, self.Ny//3].T).to("cpu")))
                #data.append(copy.deepcopy((Z0*self.ex0.T).to("cpu")))
                data.append(copy.deepcopy((self.X[3*self.Nx-1:4*self.Nx,:].T).to("cpu")))
            
        
        return data_time, data
    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(data[0],vmin = -1e-16, vmax = 1e-16)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()








# dx = 1e-10 # m
# dy = 0.125e-9# ms

# Sy = 0.8 # !Courant number, for stability this should be smaller than 1
# dt = Sy*dy/c0
# #print(dt)
# Nx = 600
# Ny = 600
# Nt = 500

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
         
    

    

