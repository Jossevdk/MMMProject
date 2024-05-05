import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
import pandas as pd
import time
import psutil


c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)

global g
g = True

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
        self.X = np.zeros((4*Nx, Ny)) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
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
        m = 4
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
        M1 = np.hstack((A1/self.dt,                np.zeros((Nx, Nx+1)),                          np.zeros((Nx, Nx-1)),     D2                   ))
        M2 = np.hstack((np.zeros((Nx, Nx-1)),       A2/self.dt,                                            D1,                  np.zeros((Nx, Nx+1))   ))
        M3 = np.hstack((-I_E/self.dt,              np.zeros((Nx-1, Nx+1)),                       np.diag(k_tot_E/self.dt+Z0*sigma_tot_E/2),              np.zeros((Nx-1, Nx+1))))
        M4 = np.hstack((np.zeros((Nx + 1, Nx-1)),  -I_H/self.dt,                                  np.zeros((Nx + 1, Nx-1)), np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)))
       
        N1 = np.hstack((A1/self.dt,                np.zeros((Nx, Nx+1)),                          np.zeros((Nx, Nx-1)),                         -D2                  ))
        N2 = np.hstack((np.zeros((Nx, Nx-1)),       A2/self.dt,                                            -D1,                                          np.zeros((Nx, Nx+1))   ))
        N3 = np.hstack((-I_E/self.dt,              np.zeros((Nx-1, Nx+1)),                       np.diag(k_tot_E/self.dt-Z0*sigma_tot_E/2),        np.zeros((Nx-1, Nx+1))))
        N4 = np.hstack((np.zeros((Nx + 1, Nx-1)),  -I_H/self.dt,                                  np.zeros((Nx + 1, Nx-1)),                      np.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2)))
       
        M = np.vstack((M1, M2, M3, M4))
        
        self.M_inv = np.linalg.inv(M)
        self.N = np.vstack((N1, N2, N3, N4))
        self.M_N = self.M_inv@self.N

        #explicit part
        self.ex = np.zeros((Nx+1, Ny+1))

        self.Y =   np.vstack((np.zeros((self.Nx, self.Ny)),self.A2@(self.ex[:, 1:] - self.ex[:, :-1])/self.dy, np.zeros((self.Nx-1, self.Ny)), np.zeros((self.Nx+1, self.Ny))))
         
        

        self.Betax_min = np.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2)
        self.Betay_min = np.eye(Nx+1)/self.dt
        self.Betaz_min = np.eye(Nx+1)/self.dt
        self.Betax_plus = np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)
        self.Betay_plus_inv = np.linalg.inv(np.eye(Nx+1)/self.dt)
        self.Betaz_plus_inv = np.linalg.inv(np.eye(Nx+1)/self.dt)

    def explicit(self):
        # self.ex2old = copy.deepcopy(self.ex2)
        # self.ex1old = copy.deepcopy(self.ex1)
        
        self.ex[:,1:-1] = self.ex[:,1:-1] + self.dt/(self.dy)*(self.X[3*self.Nx-1:4*self.Nx,1:] - self.X[3*self.Nx-1:4*self.Nx,:-1])
        
        
  

    def implicit(self, n, source):
        self.Y[self.Nx:2*self.Nx , :] = self.A2@(self.ex[:, 1:] - self.ex[:, :-1])/self.dy
        self.Y[self.Nx + int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)
        
        self.X = self.M_N@self.X + self.M_inv@(self.Y)
      
    def Update(self,n, source):
        self.implicit(n, source)
        self.explicit()
        return 

    def calculate(self, Nt, source):
        data_time = []
        data = []
        tracker = []

        for n in range(0, Nt):
            self.implicit(n, source)
            self.explicit()
            data_time.append(self.dt*n)
            #data.append(copy.deepcopy((Z0*self.ex0.T)))
            data.append((Z0*self.ex.T))
            #data.append(copy.deepcopy((self.X[3*self.Nx-1:4*self.Nx,:].T)))
            tracker.append(copy.deepcopy(Z0*self.ex[self.Nx//4,self.Ny//2]))
                        
        return data_time, data, tracker
    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source
        cax = ax.imshow(data[3],vmin = -1e-12, vmax = 1e-12)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()



dx = 1e-10 # m
dy = 0.125e-9# ms

Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0
#print(dt)
Nx = 300
Ny = 300
Nt = 400

pml_nl = 20
pml_kmax = 4
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)


xs = Nx*dx/2 
ys = Ny*dy/2

tc = dt*Nt/4
#print(tc)
sigma = tc/10
source = Source(xs, ys, 1, tc, sigma)


scheme = UCHIE(Nx, Ny, dx, dy, dt, pml_kmax = pml_kmax, pml_nl = pml_nl)
start_time = time.time()

data_time, data, tracker = scheme.calculate(Nt, source)


plt.plot(data_time, tracker)

process = psutil.Process()
print("Memory usage:", process.memory_info().rss) # print memory usage
print("CPU usage:", process.cpu_percent()) # print CPU usage

end_time = time.time()


print("Execution time: ", end_time - start_time, "seconds")

scheme.animate_field(data_time, data)
