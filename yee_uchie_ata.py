import numpy as np
from scipy.constants import mu_0 as mu0
from scipy.constants import epsilon_0 as eps0
import scipy.constants as ct
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import pandas as pd
import matplotlib.patches as patch


class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma
           
    def J(self, t):
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))



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



###### The Yee + Uchie scheme
###### ! There can be a mistake with mu_0 since assigment is given for H_z while paper for B_z
class Yee_UCHIE:

    def __init__(self, Nx, Ny, Nt, dx, dy, dt, Ly, nx, recorders=None):

        print('hello')


    def initialize(self, Nx, Ny, Nt, dx, dy, dt, Ly, n, N_sub, recorders=None):

        nx = n*N_sub
        ny = int(Ly//dy) # @ ny is the grid size of the UCHIE region
        dx_f = dx/n # @ The dx of your subgrid/UCHIE region


        # Capital letters are used for the fields in the YEE
        Ex = np.zeros((Nx+1, Ny+1))
        Ey = np.zeros((Nx, Ny))
        Bz = np.zeros((Nx+1, Ny))

        # small letters used for fields in UCHIE
        X1 = np.zeros((2*nx+2, ny))
        ex1 = np.zeros((nx+1, ny+1))
        X2 = np.zeros((2*nx+2, ny))
        ex2 = np.zeros((nx+1, ny+1))
        #@ ey = X[:nx+1, :]
        #@ bz = X[nx+1:, :]


        #Initialize the matrices for the UCHIE calculation, see eq (26)
        # ! It is even possible to just make the A_D and A_I matrix bigger instead of artificialy adding a row for the stitching
        A_D = (np.diag(-1 * np.ones(nx+1), 0) + np.diag(np.ones(nx), 1))
        A_I = (np.diag(1 * np.ones(nx+1), 0) + np.diag(np.ones(nx), 1))


        M1_2 = np.hstack((A_D/dx_f, A_I/dt))
        M1_4 = np.hstack((eps0*A_I[:-1,:]/dt, A_D[:-1,:]/(mu0*dx_f)))
        
        M1_1 = np.zeros(2*nx+2) # Stitching left interface
        M1_1[0] = 1/dx_f
        M1_1[nx+1] = 1/dt

        M1 = np.vstack((M1_1, M1_2, M1_4))
        M1_inv = np.linalg.inv(M1)
        

        M2_2 = np.hstack((-1/dx_f*A_D, 1/dt*A_I))
        M2_4 = np.hstack((eps0/dt*A_I[:-1, :], -1/(mu0*dx_f)*A_D[:-1, :]))

        M2_1 = np.zeros(2*nx+2) # Stitching left interface
        M2_1[0] = -1/dx_f
        M2_1[nx+1] = 1/dt

        M2 = np.vstack((M2_1, M2_2, M2_4))


        return M1_inv, M2, Ex, Ey, Bz, X1, X2, ex1, ex2, nx, ny, dx_f  
    



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
    def implicit_update(self, X, ex, Ex, Ey, Bz, Bz_old, x1, x2, y1, y2, nx, ny, n, N_sub):
        Y = ex[:-1, 1:] + ex[1:, 1:] - ex[:-1, :-1] - ex[1:, :-1]
            
        U_left = 1/dy*(ex[0, 1: ] + Ex[x1-1, y1+1:y2+1] - ex[0, :-1] - Ex[x1-1, y1:y2])  +  1/dx*(Ey[x1-1, y1:y2] + Ey[x1-2, y1:y2])  -  1/dt*(Bz[x1-1, y1:y2] - Bz_old[x1-1, y1:y2]) # UCHIE stitching left interface
        U_right = 1/dy*(ex[-1, 1: ] + Ex[x2+1, y1+1:y2+1] - ex[-1, :-1] - Ex[x2+1, y1:y2])  -  1/dx*(Ey[x2, y1:y2] + Ey[x2+1, y1:y2])  -  1/dt*(Bz[x2+1, y1:y2] - Bz_old[x2+1, y1:y2]) # UCHIE stitching right interface

        X = M1_inv@M2@X + M1_inv@np.vstack((U_left, Y/dy, U_right, np.zeros((nx, ny))))

        ey = X[:nx+1, :]
        bz = X[nx+1:, :]


        ### Field update in UCHIE region ex explicit ###
        ex[:, 1:-1] = ex[:, 1:-1]  +  dt/(mu0*eps0*dy) * (bz[:, 1:] - bz[:, :-1])

        A_pol = self.create_interpolation_matrix(n, nx, N_sub)
        #$ test = pd.DataFrame(A_pol)
        #$ test.columns = ['']*test.shape[1]
        #$ print(test.to_string(index=False))
        ex[:, -1] = ex[:, -1]  +  dt/(mu0*eps0*dy) * (A_pol @ Bz[x1:x2+1, y2] - bz[:, -1]) # Stitching upper interface @ Uchie
        ex[:, 0] = ex[:, 0]  -  dt/(mu0*eps0*dy) * (A_pol @ Bz[x1:x2+1, y1-1] - bz[:, 0]) # Stitching down interface @ Uchie


        return X, ex, ey, bz
    


    def stitching_B(self, x1, x2, y1, y2, Bz, ex, n):
        Bz[x1: x2+1, y1:y2] = 0 # Set the B fields in the UCHIE region to zero, in order not the double count in the updates
        Bz[x1:x2+1, y2] = Bz[x1:x2+1, y2]  -  dt/dy * ex[::n, -1] # Stitching upper interface #TODO see course notes for more accurate
        Bz[x1:x2+1, y1-1] = Bz[x1:x2+1, y1-1]  +  dt/dy * ex[::n, 0]  # Stitching lower interface #TODO see course notes for more accurate
        return Bz
    


    def stitching_E(self, x1, x2, y1, y2, Ex, Ey, bz):
        Ey[x1-1,y1:y2] = Ey[x1-1,y1:y2]  -  dt/(dx*mu0*eps0) * bz[0, :] # stiching left interface
        Ey[x2,y1:y2] = Ey[x2,y1:y2]  +  dt/(dx*mu0*eps0) * bz[-1, :] # stiching right interface

        Ex[x1:x2+1, y1:y2+1] = 0 # Fields in the UCHIE region set zero to avoid double counting
        Ey[x1:x2, y1:y2] = 0 # Fields in the UCHIE region set zero to avoid double counting
        
        return Ex, Ey




    # Procedure from the paper is being followed
    def calculate_fields(self, dx, dx_f, dy, dt, Ny, N_sub, n, nx, ny, Ex, Ey, Bz, X1, X2, ex1, ex2, M1_inv, M2, x_sub1, x_sub2, Nt, source):

        data_yee = []
        data_uchie1 = []
        data_uchie2 = []
        data_time = []
        
        #@ x_sub is the index where the subgridding is happening, start counting from 0

        x_sub1 = int(round(x_sub1/dx))
        x_sub2 = int(round(x_sub2/dx))

        y1 = (Ny-ny)//2 #@ The last index of just under the UCHIE in the Yee region
        y2 = y1 + ny #@ The first index of just above the UCHIe in the Yee region

        x1 = x_sub1
        x2 = x1 + N_sub
        x3 = x_sub2
        x4 = x3 + N_sub

        xs = int(round(source.x/dx))
        ys = int(round(source.y/dy))



        for time_step in range(0, Nt):

            print(time_step)

            ### field in Yee-region update ###
            Bz_old = Bz # TODO only the values at the interface should be saved see eq (30) of paper
            Bz[1:-1, :] = Bz_old[1:-1,:]  +  dt/dy * (Ex[1:-1, 1:] - Ex[1:-1, :-1])  -  dt/dx * (Ey[1:, :] - Ey[:-1, :]) # Update b field for left side of the uchie grid
            Bz[xs, ys] -= dt*source.J(time_step*dt) #? check if the source is added on the correct location also do we need to multiply with dx dy?

            Bz = self.stitching_B(x1, x2, y1, y2, Bz, ex1, n)
            Bz = self.stitching_B(x3, x4, y1, y2, Bz, ex2, n)


            ### Field update in UCHIE region updated, bz and ey with implicit ###
            #! I followed the stitching procedure from the paper, the stitching from the syllabus is little bit different
            X1, ex1, ey1, bz1 = self.implicit_update(X1, ex1, Ex, Ey, Bz, Bz_old, x1, x2, y1, y2, nx, ny, n, N_sub)
            X2, ex2, ey2, bz2 = self.implicit_update(X2, ex2, Ex, Ey, Bz, Bz_old, x3, x4, y1, y2, nx, ny, n, N_sub)

            
            ### Update Ex and Ey in the Yee region ###
            Ex[1:-1, 1:-1] = Ex[1:-1, 1:-1]  +  dt/(dy*mu0*eps0) * (Bz[1:-1,1:] - Bz[1:-1,:-1])
            Ey = Ey  -  dt/(dx*mu0*eps0) * (Bz[1:,:] - Bz[:-1,:])

            Ex, Ey = self.stitching_E(x1, x2, y1, y2, Ex, Ey, bz1)
            Ex, Ey = self.stitching_E(x3, x4, y1, y2, Ex, Ey, bz2)


            ### Save the data's ###
            data_yee.append(copy.deepcopy(Ey.T))
            data_uchie1.append(copy.deepcopy(ey1.T))
            data_uchie2.append(copy.deepcopy(ey2.T))
            data_time.append(time_step*dt)


        return data_time, data_yee, data_uchie1, data_uchie2



    def animate_field(self, t, data_yee, data_uchie1, data_uchie2, dx, dy, dt, source, x1, x2, y1, y2, x3, x4):

        
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        ax.set_xlim(0, (Nx+1)*dx)
        ax.set_ylim(0, (Ny+1)*dy)

        label = "Field"

        xs = source.x
        ys = source.y
        
        # v = source.J0*dt * 0.05
        v = 1e-13

        ax.plot(xs, ys+0.5*dy, color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(data_yee[0], vmin = -v, vmax = v, origin='lower', extent = [0, (Nx+1)*dx, dy/2, Ny*dy])
        ax.set_title("T = 0")

        subgrid1 = [x1, x2, y1, y2]
        scax1 = ax.imshow(data_uchie1[0], vmin=-v, vmax=v, origin='lower', extent=subgrid1)
        rect1=patch.Rectangle((x1, y1),x2-x1, y2-y1, alpha = 0.05, facecolor="grey", edgecolor="black")
        ax.add_patch(rect1)

        subgrid2 = [x3, x4, y1, y2]
        scax2 = ax.imshow(data_uchie2[0], vmin=-v, vmax=v, origin='lower', extent=subgrid2)
        rect2=patch.Rectangle((x3, y1),x4-x3, y2-y1, alpha = 0.05, facecolor="grey", edgecolor="black")
        ax.add_patch(rect2)

        def animate_frame(i):
            cax.set_array(data_yee[i])
            scax1.set_array(data_uchie1[i])
            scax2.set_array(data_uchie2[i])
            ax.set_title("T = " + str(t[i]))#"{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data_yee)), interval=20)
        plt.show()


            
            
 
        



########## Fill in the parameters here ################
Nx = 500
Ny = 500
Nt = 500

dx = 0.25e-10 # m
dy = 0.25e-10 # ms
courant = 0.9 # !Courant number, for stability this should be smaller than 1
dt = courant * 1/(np.sqrt(1/dx**2 + 1/dy**2)*ct.c)

Ly = Ny/2*dy
n = 5 #@ Subgridding in one grid
N_sub = 40 #@ How much grid you want to take to subgrid

x_sub1 = Nx//2*dx #@ The locationwhere the first subgridding should happen
x_sub2 = 1.5*Nx//2*dx #@ The locationwhere the first subgridding should happen


#create the source
xs = 2/5*Nx*dx
ys = Ny/2*dy
tc = dt*Nt/4
sigma = tc/10
source = Source(xs, ys, 1, tc, sigma)

test = Yee_UCHIE(Nx, Ny, Nt, dx, dy, dt, Ly, n, recorders=None)
M1_inv, M2, Ex, Ey, Bz, X1, X2, ex1, ex2, nx, ny, dx_f = test.initialize(Nx, Ny, Nt, dx, dy, dt, Ly, n, N_sub)
data_time, data_yee, data_uchie1, data_uchie2 = test.calculate_fields(dx, dx_f, dy, dt, Ny, N_sub, n, nx, ny, Ex, Ey, Bz, X1, X2, ex1, ex2, M1_inv, M2, x_sub1, x_sub2, Nt, source)

x1 = x_sub1
x2 = x1 + N_sub*dx
x3 = x_sub2
x4 = x3 + N_sub*dx
y1 = (Ny-ny)//2 * dy #@ The last index of just under the UCHIE in the Yee region
y2 = y1 + ny * dy #@ The first index of just above the UCHIe in the Yee region
test.animate_field(data_time, data_yee, data_uchie1, data_uchie2, dx, dy, dt, source, x1, x2, y1, y2, x3, x4)