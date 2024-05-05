import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation
import time
import psutil
#For the QM part, we require a simple 1D FDTD scheme

# from uchie_FAST import Source, UCHIE
# from PML_uchie import Source, UCHIE

#EM and QM classes containing all the necessary update function
import EM_update as EM
import QM_update as QM

eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass
q = ct.elementary_charge
c0 = ct.speed_of_light 


Z0 = np.sqrt(mu0/eps0)




#### Coupling ####

class coupled:
    def __init__(self, EMsource, EMscheme, QMscheme, Nt):
        self.EMsource = EMsource
        self.EMscheme = EMscheme
        self.QMscheme= QMscheme
        self.Nt = Nt


    def calcwave(self):
        for n in range (self.Nt):
            # EMscheme.implicit(n, EMsource, QMscheme.J)
            # EMscheme.explicit()
            Efield = 5*1e16*EMscheme.Update(n,EMsource,QMscheme.J )
            #E = copy.deepcopy(Efield[2*Nx//4,:])
            E = Efield[2*Nx//4,:]
            #efield = EMscheme.X[:] #add the correct selection here
            QMscheme.update(Efield,n)



################################################
#all the input for EM part
dx = 1e-10 # m
dy = 0.125e-9# ms

Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0

Nx = 400
Ny = 400
Nt = 400

pml_nl = 20
pml_kmax = 4
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)


xs = Nx*dx/4 
ys = Ny*dy/2


J0 = 1
tc = dt*Nt/4
sigma = tc/10
QMxpos = Nx*dx/2  #this is where the quantum sheet is positioned

EMsource = EM.Source(xs, ys, 1, tc, sigma)
EMscheme = EM.UCHIE(Nx, Ny, dx, dy, dt, QMxpos,pml_kmax = pml_kmax, pml_nl = pml_nl)


#############################################################
#QM input parameters


N = 30000 #particles/m2

order = 'fourth'
omega = 50e12 #[rad/s]
alpha = 0
potential = QM.Potential(m,omega, Ny, dy)
#potential.V()

QMscheme = QM.QM(order,Ny,dy, dt, hbar, m, q, alpha, potential, omega)

#############################################################
#start the coupled simulation
Nt = 200

coupledscheme = coupled(EMsource,EMscheme, QMscheme, Nt)

coupledscheme.calcwave()


# qm = coupled(order, Nx,Ny, dx, dy,dt, pml_kmax, pml_nl, J0, xs, ys, tc, sigma)

# start_time = time.time()


# res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, alpha, order,N)
# prob = res[3]
# probsel = prob[::100]


# process = psutil.Process()
# print("Memory usage:", process.memory_info().rss/(1024*1024), 'MB') # print memory usage
# print("CPU usage:", process.cpu_percent()) # print CPU usage

# end_time = time.time()


# print("Execution time: ", end_time - start_time, "seconds")

# qm.animate( dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
# qm.animate_field(res[0], res[5])



# qm.heatmap(dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
