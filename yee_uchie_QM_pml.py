import numpy as np
from scipy.constants import mu_0 as mu0
from scipy.constants import epsilon_0 as eps0
import scipy.constants as ct
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import pandas as pd
import matplotlib.patches as patch

import QM_update as QM

eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass*0.15
q = -ct.elementary_charge
c0 = ct.speed_of_light 


Z0 = np.sqrt(mu0/eps0)


class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma
           
    def J(self, t):
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))

def genKy(Ny, Nx, pml_nl, m, dy):
    Ky = np.zeros((Nx+1, Ny+1))
    kmax = -np.log(np.exp(-16))*(m+1)/(2*np.sqrt(ct.mu_0/ct.epsilon_0)*pml_nl*dy)
    for iy in range(0, pml_nl):
        Ky[:,iy] = kmax*((pml_nl-1-iy)/pml_nl)**m
        Ky[:, -iy-1] = kmax*((pml_nl-1-iy)/pml_nl)**m
    return Ky

def genKx(Ny, Nx, pml_nl, m, dx):
    Kx = np.zeros((Nx+1, Ny+1))
    kmax = -np.log(np.exp(-16))*(m+1)/(2*np.sqrt(ct.mu_0/ct.epsilon_0)*pml_nl*dx)
    for ix in range(0, pml_nl):
        Kx[ix,:] = kmax*((pml_nl-1-ix)/pml_nl)**m
        Kx[-ix-1,:] = kmax*((pml_nl-1-ix)/pml_nl)**m
    return Kx

class Recorder:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data = []          # data of the field will be added in this list
        self.data_time = []     # data of the time will be added in this list

    # adding the measurement of the field to the data
    def save_data(self, field, t):
        self.data.append(field) # appending a measurement to the list
        self.data_time.append(t)



###### The Yee + Uchie scheme ######
###### ! There can be a mistake with mu_0 since assigment is given for H_z while paper for B_z 
class Yee_UCHIE:
    def __init__(self, Nx, Ny, Nt, dx, dy, dt, Ly, n, N_sub, NYQM, x_sub1, x_sub2, eps, mu, source, QMscheme1, QMscheme2, QMxpos1, QMxpos2,recorders=[]):
        
        self.mu = mu
        self.eps = eps

        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        
        self.dx = dx
        self.dx_f = dx/n # @ The dx of your subgrid/UCHIE region
        self.dy = dy
        self.dt = dt

        self.Ly = Ly

        self.n = n
        self.N_sub = N_sub
        self.nx = n*N_sub
        self.ny = int(Ly/dy) # @ ny is the grid size of the UCHIE region

        self.NyQM = NYQM

        self.source = source
        self.xs = int(round(source.x/dx))
        self.ys = int(round(source.y/dy))

        self.QMscheme1 = QMscheme1
        self.QMscheme2 = QMscheme2

        self.QMxpos1 = QMxpos1
        self.QMxpos2 = QMxpos2

        self.recorders = recorders

        self.x_sub1 = x_sub1
        self.x_sub2 = x_sub2
        self.x1 = self.x_sub1
        self.x2 = self.x1 + N_sub*dx
        self.x3 = self.x_sub2
        self.x4 = self.x3 + N_sub*dx
        self.y1 = (Ny-self.ny)//2 * dy #@ The last index of just under the UCHIE in the Yee region
        self.y2 = self.y1 + self.ny * dy #@ The first index of just above the UCHIe in the Yee region


        #@ x_sub is the index where the subgridding is happening, start counting from 0

        self.x_sub1 = int(round(x_sub1/dx))
        self.x_sub2 = int(round(x_sub2/dx))

        # Capital letters are used for the fields in the YEE
        pml_nl = 20
        pml_m = 4
        self.Ex = np.zeros((Nx+1, Ny+1))
        self.Ey = np.zeros((Nx, Ny))
        self.Bz = np.zeros((Nx+1, Ny))
        self.Bzx = np.zeros((Nx+1, Ny))
        self.Bzy = np.zeros((Nx+1, Ny))
        Ky = genKy(Ny, Nx, pml_nl, pml_m, dy)
        Kx = genKx(Ny, Nx, pml_nl, pml_m, dx)
        print(Kx, "\n")
        self.KxE = (2*np.full((Nx+1, Ny+1), ct.epsilon_0) - Kx*dt)/(2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Kx*dt)
        self.KyE = (2*np.full((Nx+1, Ny+1), ct.epsilon_0) - Ky*dt)/(2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Ky*dt)
        self.KxB = (2*np.full((Nx+1, Ny+1), ct.mu_0) - ct.mu_0*Kx*dt/ct.epsilon_0)/(2*np.full((Nx+1, Ny+1), ct.mu_0) + ct.mu_0*Kx*dt/ct.epsilon_0)
        self.KyB = (2*np.full((Nx+1, Ny+1), ct.mu_0) - ct.mu_0*Ky*dt/ct.epsilon_0)/(2*np.full((Nx+1, Ny+1), ct.mu_0) + ct.mu_0*Ky*dt/ct.epsilon_0)
        self.KxEB = (2*dt)/((2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Kx*dt)*dx*ct.mu_0)
        self.KyEB = (2*dt)/((2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Ky*dt)*dy*ct.mu_0)
        self.KxBE = (2*ct.mu_0*dt)/((2*np.full((Nx+1, Ny+1), ct.mu_0) +ct.mu_0*Kx*dt/ct.epsilon_0)*dx)
        self.KyBE = (2*ct.mu_0*dt)/((2*np.full((Nx+1, Ny+1), ct.mu_0) + ct.mu_0*Ky*dt/ct.epsilon_0)*dy)
        
        # small letters used for fields in UCHIE
        self.X1 = np.zeros((2*self.nx+2, self.ny))
        self.ex1 = np.zeros((self.nx+1, self.ny+1))
        self.X2 = np.zeros((2*self.nx+2, self.ny))
        self.ex2 = np.zeros((self.nx+1, self.ny+1))
        #@ ey = X[:nx+1, :]
        #@ bz = X[nx+1:, :]

        self.data_yee = []
        self.data_uchie1 = []
        self.data_uchie2 = []
        self.data_time = []

        


        #Initialize the matrices for the UCHIE calculation, see eq (26)
        # ! It is even possible to just make the A_D and A_I matrix bigger instead of artificialy adding a row for the stitching
        A_D = np.diag(-1 * np.ones(self.nx+1), 0) + np.diag(np.ones(self.nx), 1)
        A_D = A_D[:-1, :]


        A_I = np.zeros((self.nx, self.nx + 1))
        np.fill_diagonal(A_I, 1)
        np.fill_diagonal(A_I[:,1:], 1)

        M1_1 = np.zeros(2*self.nx+2) # extra row for Stitching left interface
        M1_1[0] = 1/dx
        M1_1[self.nx+1] = 1/dt

        M1_2 = np.hstack((A_D/self.dx_f, A_I/dt))

        M1_3 = np.zeros(2*self.nx+2) # The row to stitch the right interface with Yee, see eq (30)
        M1_3[self.nx] = -1/dx
        M1_3[-1] = 1/dt

        M1_4 = np.hstack((eps0*A_I/dt, A_D/(mu*self.dx_f)))

        M1 = np.vstack((M1_1, M1_2, M1_3, M1_4))
        self.M1_inv = np.linalg.inv(M1)
        
        M2_1 = np.zeros(2*self.nx+2) # Stitching left interface
        M2_1[0] = -1/dx
        M2_1[self.nx+1] = 1/dt

        M2_2 = np.hstack((-1/self.dx_f*A_D, 1/dt*A_I))

        M2_3 = np.zeros(2*self.nx+2) # The row to stich the right interface with Yee, see eq(30)
        M2_3[self.nx] = 1/dx
        M2_3[-1] = 1/dt
        
        M2_4 = np.hstack((eps/dt*A_I, -1/(mu*self.dx_f)*A_D))

        self.M2 = np.vstack((M2_1, M2_2, M2_3, M2_4))

    

    # Procedure from the paper is being followed
    def calculate_fields(self):

        x1 = int(round(self.x1/self.dx))
        x2 = int(round(self.x2/self.dx))
        x3 = int(round(self.x3/self.dx))
        x4 = int(round(self.x4/self.dx))
        y1 = int(round(self.y1/self.dy))
        y2 = int(round(self.y2/self.dy))

        for time_step in range(0, Nt):

            print(time_step)

            ### field in Yee-region update ###
            Bz_old = self.Bz # TODO only the values at the interface should be saved see eq (30) of paper
            #self.Bz[1:-1, :] = Bz_old[1:-1,:]  +  self.dt/self.dy * (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1])  -  self.dt/self.dx * (self.Ey[1:, :] - self.Ey[:-1, :]) # Update b field for left side of the uchie grid
            self.Bzy[1:-1, :] = (self.KyB[1:-1, :-1] + self.KyB[1:-1, 1:])/2 * self.Bzy[1:-1, :]  +  (self.KyBE[1:-1, :-1] + self.KyBE[1:-1, 1:])/2 * (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1])   # Update b field for right side of the uchie grid
            self.Bzx[1:-1, :] = (self.KxB[1:-1, :-1] + self.KxB[1:-1, 1:])/2 * self.Bzx[1:-1, :]  - (self.KyBE[1:-1, :-1] + self.KyBE[1:-1, 1:])/2 * (self.Ey[1:, :] - self.Ey[:-1, :]) # Update b field for right side of the uchie grid
            self.Bzy[self.xs, self.ys] -= dt*source.J(time_step*self.dt)/2#? check if the source is added on the correct location also do we need to multiply with dx dy?
            self.Bzx[self.xs, self.ys] -= dt*source.J(time_step*self.dt)/2#? check if the source is added on the correct location also do we need to multiply with dx dy?

            
            self.Bzx = self.stitching_B(x1, x2, y1, y2, self.Bzx, self.ex1, self.n, self.dt, self.dy) # Stitching first subgrid
            self.Bzy = self.stitching_B(x1, x2, y1, y2, self.Bzy, self.ex1, self.n, self.dt, self.dy) # Stitching first subgrid
            self.Bzx = self.stitching_B(x3, x4, y1, y2, self.Bzx, self.ex2, self.n, self.dt, self.dy) # Stitching second subgrid
            self.Bzy = self.stitching_B(x3, x4, y1, y2, self.Bzy, self.ex2, self.n, self.dt, self.dy) # Stitching second subgrid
            self.Bz = self.Bzx + self.Bzy
            #Update QM schemes
            slice = int(1/2*(self.ny-self.NyQM))
            self.QMscheme1.update(1e20*self.X1[self.QMxpos1,slice:-slice],time_step)
            self.QMscheme2.update(1e20*self.X2[self.QMxpos2,slice:-slice],time_step)


            ### Field update in UCHIE region updated, bz and ey with implicit ###
            self.X1, self.ex1, self.ey1, self.bz1 = self.implicit_update(self.X1, self.ex1, self.Ex, self.Ey, self.Bz, Bz_old, x1, x2, y1, y2, self.nx, self.ny, self.n, self.N_sub, self.dx_f, self.dx, self.dy, self.dt, self.M1_inv, self.M2, self.eps, self.mu, QMscheme1.J, QMxpos1)
            self.X2, self.ex2, self.ey2, self.bz2 = self.implicit_update(self.X2, self.ex2, self.Ex, self.Ey, self.Bz, Bz_old, x3, x4, y1, y2, self.nx, self.ny, self.n, self.N_sub, self.dx_f, self.dx, self.dy, self.dt, self.M1_inv, self.M2, self.eps, self.mu, QMscheme2.J, QMxpos2)

            
            ### Update Ex and self.Ey in the Yee region ###
            self.Ex[1:-1, 1:-1] = self.KyE[1:-1, 1:-1]*self.Ex[1:-1, 1:-1]  +  self.KyEB[1:-1,1:-1] * (self.Bz[1:-1,1:] - self.Bz[1:-1,:-1])
            self.Ey = (self.KxE[:-1, :-1] +self.KxE[1:, :-1] +self.KxE[:-1, 1:] +self.KxE[1:, 1:])/4 *self.Ey  -  (self.KxEB[:-1,:-1] +self.KxEB[1:,:-1]+self.KxEB[:-1,1:]+self.KxEB[1:,1:])/4 * (self.Bz[1:,:] - self.Bz[:-1,:])

            self.Ex, self.Ey = self.stitching_E(x1, x2, y1, y2, self.Ex, self.Ey, self.bz1, self.dx, self.dt, self.mu, self.eps) # stitching first subgrid
            self.Ex, self.Ey = self.stitching_E(x3, x4, y1, y2, self.Ex, self.Ey, self.bz2, self.dx, self.dt, self.mu, self.eps) # stitching second subgrid


            ### Save the data's ###
            self.data_yee.append(copy.deepcopy(self.Bz.T))
            self.data_uchie1.append(copy.deepcopy(self.bz1.T))
            self.data_uchie2.append(copy.deepcopy(self.bz2.T))
            self.data_time.append(time_step*self.dt)



    # Create interpolation matrix for the stitching
    def create_interpolation_matrix(self, n, nx, N_sub):
        A_pol = np.zeros((nx+1, N_sub+1))
        for i in range(N_sub):
            A_1 = np.arange(n+1)/n 
            A_2 = A_1[::-1]
            A = np.vstack((A_2, A_1)).T #@ The interpolation matrix, see notes
            A = A[:-1,:]
            A_pol[i*n:(i+1)*n, i:i+2] = A
        A_pol[-1, -1] = 1
        return A_pol
    


    ### Field update in UCHIE region updated, bz and ey with implicit ###
    #! I followed the stitching procedure from the paper, the stitching from the syllabus is little bit different
    def implicit_update(self, X, ex, Ex, Ey, Bz, Bz_old, x1, x2, y1, y2, nx, ny, n, N_sub, dx_f, dx, dy, dt, M1_inv, M2, eps, mu, JQM, QMxpos):
        Y = ex[:-1, 1:] + ex[1:, 1:] - ex[:-1, :-1] - ex[1:, :-1]
            
        U_left = 1/dy*(ex[0, 1: ] + Ex[x1-1, y1+1:y2+1] - ex[0, :-1] - Ex[x1-1, y1:y2])  +  1/dx*(Ey[x1-1, y1:y2] + Ey[x1-2, y1:y2])  -  1/dt*(Bz[x1-1, y1:y2] - Bz_old[x1-1, y1:y2]) # UCHIE stitching left interface
        U_right = 1/dy*(ex[-1, 1: ] + Ex[x2+1, y1+1:y2+1] - ex[-1, :-1] - Ex[x2+1, y1:y2])  -  1/dx*(Ey[x2, y1:y2] + Ey[x2+1, y1:y2])  -  1/dt*(Bz[x2+1, y1:y2] - Bz_old[x2+1, y1:y2]) # UCHIE stitching right interface

        X = M1_inv@M2@X + M1_inv@np.vstack((U_left, Y/dy, U_right, np.zeros((nx, ny))))

        ey = X[:nx+1, :]
        bz = X[nx+1:, :]


        ### Field update in UCHIE region ex explicit ###
        ex[:, 1:-1] = ex[:, 1:-1]  +  dt/(mu*eps*dy) * (bz[:, 1:] - bz[:, :-1])

        A_pol = self.create_interpolation_matrix(n, nx, N_sub)
        #$ test = pd.DataFrame(A_pol)
        #$ test.columns = ['']*test.shape[1]
        #$ print(test.to_string(index=False))
        ex[:, -1] = ex[:, -1]  +  dt/(mu*eps*dy) * (A_pol @ Bz[x1:x2+1, y2] - bz[:, -1]) # Stitching upper interface @ Uchie
        ex[:, 0] = ex[:, 0]  -  dt/(mu*eps*dy) * (A_pol @ Bz[x1:x2+1, y1-1] - bz[:, 0]) # Stitching down interface @ Uchie

        slice = int(1/2*(self.ny-self.NyQM))
        Y[QMxpos, slice :-slice]+= -2 * (1 / Z0) * JQM

        return X, ex, ey, bz
    


    def stitching_B(self, x1, x2, y1, y2, Bz, ex, n, dt, dy):
        Bz[x1: x2+1, y1:y2] = 0 # Set the B fields in the UCHIE region to zero, in order not the double count in the updates
        Bz[x1:x2+1, y2] = Bz[x1:x2+1, y2]  -  dt/dy * ex[::n, -1]/2 # Stitching upper interface #TODO see course notes for more accurate
        Bz[x1:x2+1, y1-1] = Bz[x1:x2+1, y1-1]  +  dt/dy * ex[::n, 0]/2  # Stitching lower interface #TODO see course notes for more accurate
        return Bz
    


    def stitching_E(self, x1, x2, y1, y2, Ex, Ey, bz, dx, dt, mu, eps):
        Ey[x1-1,y1:y2] = Ey[x1-1,y1:y2]  -  dt/(dx*mu*eps) * bz[0, :] # stiching left interface
        Ey[x2,y1:y2] = Ey[x2,y1:y2]  +  dt/(dx*mu*eps) * bz[-1, :] # stiching right interface

        Ex[x1:x2+1, y1:y2+1] = 0 # Fields in the UCHIE region set zero to avoid double counting
        Ey[x1:x2, y1:y2] = 0 # Fields in the UCHIE region set zero to avoid double counting
        
        return Ex, Ey



    def animate_field(self):

        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        ax.set_xlim(0, (Nx+1)*dx)
        ax.set_ylim(0, (Ny+1)*dy)

        label = "Field"

        xs = self.source.x
        ys = self.source.y
        
        # v = source.J0*dt * 0.05
        v = 3e-22

        ax.plot(xs, ys+0.5*dy, color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(self.data_yee[0], vmin = -v, vmax = v, origin='lower', extent = [0, (self.Nx+1)*self.dx, self.dy/2, self.Ny*self.dy])
        ax.set_title("T = 0")

        subgrid1 = [self.x1, self.x2, self.y1, self.y2]
        scax1 = ax.imshow(self.data_uchie1[0], vmin=-v, vmax=v, origin='lower', extent=subgrid1)
        rect1=patch.Rectangle((self.x1, self.y1),self.x2-self.x1, self.y2-self.y1, alpha = 0.05, facecolor="grey", edgecolor="black")
        ax.add_patch(rect1)

        subgrid2 = [self.x3, self.x4, self.y1, self.y2]
        scax2 = ax.imshow(self.data_uchie2[0], vmin=-v, vmax=v, origin='lower', extent=subgrid2)
        rect2=patch.Rectangle((self.x3, self.y1),self.x4-self.x3, self.y2-self.y1, alpha = 0.05, facecolor="grey", edgecolor="black")
        ax.add_patch(rect2)

        def animate_frame(i):
            cax.set_array(self.data_yee[i])
            scax1.set_array(self.data_uchie1[i])
            scax2.set_array(self.data_uchie2[i])
            ax.set_title("T = " + str(self.data_time[i]))#"{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(self.data_yee)), interval=20)
        plt.show()


            
            
 
        



########## Fill in the parameters here ################
Nx = 301
Ny = 301
Nt = 500

dx = 0.25e-10 # m
dy = 0.25e-10 # ms
courant = 0.9 # !Courant number, for stability this should be smaller than 1
dt = courant * 1/(np.sqrt(1/dx**2 + 1/dy**2)*ct.c)

Ly = 3/5*Ny*dy
n = 5 #@ numbers of Subgridding in one grid
N_sub = 30 #@ How much grid you want to take to subgrid

x_sub1 = Nx//3*dx #@ The locationwhere the first subgridding should happen
x_sub2 = 2*Nx//3*dx #@ The locationwhere the first subgridding should happen

QMxpos1 = N_sub//2
QMxpos2 = N_sub//2


NyQM = int(2*Ny/5)

#create the source
xs = 1/4*Nx*dx
ys = Ny/2*dy
tc = dt*Nt/4
sigma = tc/10
source = Source(xs, ys, 1, tc, sigma)

N = 10e7*dx #particles/m2
#NyQM = int(2*Ny/3)
order = 'fourth'
omega = 50e14 #[rad/s]
alpha = 0
potential = QM.Potential(m,omega, NyQM, dy)
#potential.V()

QMscheme1 = QM.QM(order,NyQM,dy, dt, hbar, m, q, alpha, potential, omega, N)
QMscheme2 = QM.QM(order,NyQM,dy, dt, hbar, m, q, alpha, potential, omega, N)

test = Yee_UCHIE(Nx, Ny, Nt, dx, dy, dt, Ly, n, N_sub, NyQM, x_sub1, x_sub2, eps0, mu0, source, QMscheme1 , QMscheme2, QMxpos1, QMxpos2)
test.calculate_fields()
test.animate_field()

# QMscheme1.expvalues('energy')
# QMscheme2.expvalues('energy')

