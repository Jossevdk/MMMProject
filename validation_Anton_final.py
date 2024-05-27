import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import pandas as pd
from scipy.sparse import csr_matrix
import torch as th
import time
import psutil
import numpy.fft as ft
from scipy.special import hankel2
import scipy.constants as ct


c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)


device = th.device("cuda" if th.cuda.is_available() else "cpu")




### Recorders ###
class Recorder:
    def __init__(self, x, y, field):
        self.x = x
        self.y = y
        self.data = []          # data of the field will be added in this list
        self.data_time = []     # data of the time will be added in this list
        self.field = field      # type of field int 0, 1 or 2 for e_x, e_y, and h_z

    # adding the measurement of the field to the data
    def save_data(self, field, t):
        self.data.append(field) # appending a measurement to the list
        self.data_time.append(t)



### Source ###
class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma
        self.omegamax = 3/self.sigma
        #self.omega = omega
        
    #This will call the function depending on which type of source you have    
    def J(self, t):
        #print(t)
        #return 10e7*np.cos(2*np.pi*2e7*t + 0.5)
        #print(t, self.tc, self.sigma)
        return self.J0*np.exp(-(t-self.tc)**2.0/(2*self.sigma**2.0))
    
    def J_abs(self, omega):
        return self.J0*np.sqrt(2*np.pi)*self.sigma*np.exp(-self.sigma**2*omega**2/2)
    
    # def J_sinus(self, t, omega):
    #     return self.J0*np.sin(self.omega*t)



#### UCHIE ####
class UCHIE:
    def __init__(self, Nx, Ny, dx, dy, dt, pml_kmax = None, pml_nl = None, recorders=None):
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
        self.X = th.zeros((2*Nx, Ny)).to(device) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
        self.Y = th.zeros((2*Nx, Ny)).to(device)

        self.recorders = recorders

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
        # pml_kymax = 4
        # pml_sigmay_max = (m+1)/(150*np.pi*dy)
        # pml_ky = np.array([1 + (pml_kymax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        # pml_sigmay = np.array([pml_sigmay_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        #print(np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2))
        
        M_midden1 = th.hstack((A1/self.dt, D2))
        M_midden2 = th.hstack((D1, A2/self.dt))

        M = th.vstack((M_midden1, M_midden2)).to(device)

        N_midden1 = th.hstack((A1/self.dt, -D2))
        N_midden2 = th.hstack((-D1, A2/self.dt))

        self.N = th.vstack((N_midden1, N_midden2)).to(device)

    
        
        self.M_inv = th.linalg.inv(M).to(device)
        
        self.M_N = th.mm(self.M_inv,self.N)



        #explicit part
        
        self.ex0 = th.zeros((Nx+1, Ny+1)).to(device)


        

    def explicit(self):
        
        self.ex0[:,1:-1] = self.ex0[:,1:-1] + self.dt/(self.dy)*(self.X[self.Nx-1:2*self.Nx,1:] - self.X[self.Nx-1:2*self.Nx,:-1])
        
  

    def implicit(self, n, source):
        #S_ = th.zeros((self.Nx, self.Ny))
        #S_[int(source.x/self.dx), int(source.y/self.dy)] = -2*(1/Z0)*source.J(n*self.dt/c0)
        #Y = th.vstack((th.zeros((self.Nx, self.Ny)),S_ + th.mm(self.A2, (self.ex0[:, 1:] - self.ex0[:, :-1]))/self.dy, th.zeros((self.Nx-1, self.Ny)), th.zeros((self.Nx+1, self.Ny)), th.zeros((self.Nx-1, self.Ny)) ))
        self.Y[self.Nx:2*self.Nx , :] =  th.mm(self.A2, (self.ex0[:, 1:] - self.ex0[:, :-1]))/self.dy
        self.Y[self.Nx + int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)/self.dx/self.dy#*self.dt/c0
        #S = np.zeros((5*self.Nx-1, self.Ny))
        #S[self.Nx-1 + int(source.x/self.dx), int(source.y/self.dy)] = -2*(1/Z0)*source.J(n*self.dt/c0)*self.dt/c0
        #self.X[self.Nx-1 + int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)*self.dt/c0
        
        self.X = th.mm(self.M_N, self.X )+ th.mm(self.M_inv, self.Y)
        
        # print(np.shape(self.X))

    def Update(self,n, source):
        self.implicit(n, source)
        self.explicit()
        return Z0*self.X[:self.Nx-1,:].to("cpu").numpy()

    def calculate(self, Nt, source):
        data_time = []
        data = []
        tracker = []

        for n in range(0, Nt):
            self.implicit(n, source)
            self.explicit()
            if n % 1 == 0:
                #print(n)
                data_time.append(self.dt*n)
                data.append(copy.deepcopy((Z0*self.ex0.T).to("cpu")))
                tracker.append(copy.deepcopy(self.X[self.Nx - 1 + self.Nx//3,self.Ny//3].to('cpu')))
                data.append(copy.deepcopy((self.X[self.Nx - 1:,:].T).to('cpu')))

                for recorder in self.recorders:
                    if recorder.field == 2:
                        recorder.save_data(self.X[self.Nx-1 + int(round(recorder.x/self.dx)), int(round(recorder.y/self.dy))], n*self.dt/c0)
            
        
        return data_time, data, tracker
    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(data[0],vmin = -1e-13, vmax = 1e-13)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()
    

    # TODO

    def fourier(self, Hz, w_max, ZP, rate):
        fourier_transform = np.fft.fft(Hz, n=ZP)*rate
        freq_axis = np.fft.fftfreq(ZP, rate)
        index = np.where(((freq_axis >= 0) & (freq_axis <= w_max/(2*np.pi))))
        freq_axis = freq_axis[index]
        fourier_transform = fourier_transform[index]
        return freq_axis, fourier_transform
    


    def validation(self, recorder, source, dx, dy):
        x = recorder.x
        y = recorder.y
        sigma = source.sigma

        plt.plot(recorder.data_time, recorder.data)
        plt.show()
        plt.close()

        

        omega_max = 3/sigma
        Hz = recorder.data

        padding = 100000
        freq_axis, FT = self.fourier(Hz, omega_max, padding, self.dt/c0)
        omega = freq_axis*2*np.pi
        #bb = source_1.w_max
        plt.plot(FT)
        plt.show()
        # omega = 2*np.pi*ft.rfftfreq(10000, self.dt)  # Get the frequency
        # Hz_freq = ft.rfft(Hz, 10000)

        # width = next((i for i, val in enumerate(omega) if val > omega_max), len(omega))  # Find index where omega > omega_max

        # omega = omega[:width] 
        # Hz_freq = Hz_freq[:width]
        spectralcontent = source.J0*np.sqrt(2*np.pi)*sigma*np.exp(-sigma**2*omega**2/2)
        k0 = omega/ct.c
        z = k0*np.sqrt((x - source.x)**2 + (y - source.y)**2)

        Hz_ana = -source.J0*omega*eps0/4 * hankel2(0, z)

        #print( Hz_ana[1000]/FT/spectralcontent
        plt.plot(omega, np.abs(FT/(spectralcontent)*source.J0))
        plt.plot(omega, np.abs(Hz_ana))
        plt.title("Validation magnetic field at location (" + "{:.6g}".format(x) + "m, " + "{:.6g}".format(y) + "m)")
        plt.xlabel("frequency $\omega$ [Hz]")
        plt.ylabel("$H_{z}$ [A/m]")
        plt.legend(["measured", "theoretical"])
        
        plt.show()
        plt.close()

        



dx = 0.25e-10 # m
dy = 0.25e-10# ms

Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0
#print(dt)

Nx = 400
Ny = 400
Nt = 300

pml_nl = 1
pml_kmax = 1
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)


xs = Nx*dx/2 
ys = Ny*dy/2

tc = dt*Nt/4
#print(tc)
sigma = tc/10
J_0 = 1000
recorder1 = Recorder(0.75*Nx*dx, 0.5*Ny*dy, 2)
recorders = [recorder1]
source = Source(xs, ys, J_0, tc, sigma)


scheme = UCHIE(Nx, Ny, dx, dy, dt, pml_kmax = pml_kmax, pml_nl = pml_nl, recorders = recorders)
start_time = time.time()

data_time, data, tracker = scheme.calculate(Nt, source)

#plt.plot(data_time, tracker)
process = psutil.Process()
print("Memory usage:", process.memory_info().rss) # print memory usage
print("CPU usage:", process.cpu_percent()) # print CPU usage

end_time = time.time()


print("Execution time: ", end_time - start_time, "seconds")

#scheme.animate_field(data_time, data)
scheme.validation(recorder1, source, dx, dy)