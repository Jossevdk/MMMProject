import numpy as np
from scipy.constants import mu_0 as mu0
from scipy.constants import epsilon_0 as eps0
import scipy.constants as ct
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import pandas as pd


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

    def __init__(self, Nx, Ny, Nt, dx, dy, dt, Ly, x_sub, nx, recorders=None):

        print('hello')


    def initialize(self, Nx, Ny, Nt, dx, dy, dt, Ly, x_sub, nx, recorders=None):

        #@ x_sub is the x location where your subgrid starts (integer)
        ny = int(Ly//dy) # @ ny is the grid size of the UCHIE region
        dx_f = dx/nx # @ The dx of your subgrid/UCHIE region

        # Capital letters are used for the fields in the YEE
        Ex = np.zeros((Nx+1, Ny+1))
        Ey = np.zeros((Nx, Ny))
        Bz = np.zeros((Nx+1, Ny))

        # small letters used for fields in UCHIE
        X = np.zeros((2*nx+2, ny))
        ex = np.zeros((nx+1, ny+1))
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

        # $ M1_3 = np.zeros(2*nx+2) # The row to stitch the right interface with Yee, see eq (30)
        # $ M1_3[nx] = -1/dx
        # $ M1_3[-1] = 1/dt

        M1 = np.vstack((M1_1, M1_2, M1_4))
        M1_inv = np.linalg.inv(M1)
        

        M2_2 = np.hstack((-1/dx_f*A_D, 1/dt*A_I))
        M2_4 = np.hstack((eps0/dt*A_I[:-1, :], -1/(mu0*dx_f)*A_D[:-1, :]))
        
        # $ M2_3 = np.zeros(2*nx+2) # The row to stich the right interface with Yee, see eq(30)
        # $ M2_3[nx] = -1/dx
        # $ M2_3[-1] = -1/dt

        M2_1 = np.zeros(2*nx+2) # Stitching left interface
        M2_1[0] = -1/dx_f
        M2_1[nx+1] = 1/dt

        M2 = np.vstack((M2_1, M2_2, M2_4))


        #$ A = pd.DataFrame(M1)
        #$ A.columns = ['']*A.shape[1]
        #$ print(A.to_string(index=False))
        
        #$ A = pd.DataFrame(M2)
        #$ A.columns = ['']*A.shape[1]
        #$ print(A.to_string(index=False))


        return M1_inv, M2, Ex, Ey, Bz, X, ex, ny, dx_f        



    def implicit(self, n):
        pass



    # Procedure from the paper is being followed
    def calculate_fields(self, dx, dx_f, dy, dt, Ny, nx, ny, Ex, Ey, Bz, X, ex, M1_inv, M2, x_sub, Nt, source):

        data = []
        data_time = []
        
        #@ x_sub is the index where the subgridding is happening, start counting from 0
        #$ x1 = x_sub #@ The last index of the left interface in the Yee region
        #$ x2 = x_sub+1 #@ The first index of the right interface in the Yee region
        #$ y1 = (Ny-ny)//2 #@ The last index of just under the UCHIE in the Yee region
        #$ y2 = y1 + ny-1 #@ The first index of just above the UCHIe in the Yee region

        y1 = (Ny-ny)//2 #@ The last index of just under the UCHIE in the Yee region
        y2 = y1 + ny #@ The first index of just above the UCHIe in the Yee region

        x1 = x_sub
        x2 = x1 + 1

        xs = int(round(source.x/dx))
        ys = int(round(source.y/dy))



        for n in range(0, Nt):

            print(n)

            ### field in Yee-region update ###
            Bz_old = Bz # TODO only the values at the interface should be saved see eq (30) of paper
            Bz[1:-1, :] = Bz_old[1:-1,:]  +  dt/dy * (Ex[1:-1, 1:] - Ex[1:-1, :-1])  -  dt/dx * (Ey[1:, :] - Ey[:-1, :]) # Update b field for left side of the uchie grid
            Bz[xs, ys] -= dt*source.J(n*dt) #? check if the source is added on the correct location also do we need to multiply with dx dy?
            Bz[x1: x2+1, y1:y2] = 0 # Set the B fields in the UCHIE region to zero, in order not the double count in the updates

            # Stiching
            Bz[x1, y2] = Bz[x1, y2] - dt/dy * ex[0, -1] # Stitching upper left the UCHIE #TODO see course notes for more accurate
            Bz[x2, y2] = Bz[x2, y2] - dt/dy * ex[-1, -1] # Stitching upper right the UCHIE #TODO see course notes for more accurate
            Bz[x1, y1-1] = Bz[x1, y1-1] + dt/dy * ex[0, 0]  # Stitching down left the UCHIE #TODO see course notes for more accurate
            Bz[x2, y1-1] = Bz[x2, y1-1] + dt/dy * ex[-1, 0] # Stitching upper right the UCHIE #TODO see course notes for more accurate


            ### Field update in UCHIE region updated, bz and ey with implicit ###
            #! I followed the stitching procedure from the paper, the stitching from the syllabus is little bit different
            Y = ex[:-1, 1:] + ex[1:, 1:] - ex[:-1, :-1] - ex[1:, :-1]
            #$ ey = X[:nx+1, :]
            #$ bz = X[nx+1:, :]

            #$ X = np.vstack((bz, ey))
            U_left = 1/dy*(ex[0, 1: ] + Ex[x1-1, y1+1:y2+1] - ex[0, :-1] - Ex[x1-1, y1:y2])  +  1/dx*(Ey[x1-1, y1:y2] + Ey[x1-2, y1:y2])  -  1/dt*(Bz[x1-1, y1:y2] - Bz_old[x1-1, y1:y2]) # UCHIE stitching left interface
            U_right = 1/dy*(ex[-1, 1: ] + Ex[x2+1, y1+1:y2+1] - ex[-1, :-1] - Ex[x2+1, y1:y2])  -  1/dx*(Ey[x2, y1:y2] + Ey[x2+1, y1:y2])  -  1/dt*(Bz[x2+1, y1:y2] - Bz_old[x2+1, y1:y2]) # UCHIE stitching right interface
            X = M1_inv@M2@X + M1_inv@np.vstack((U_left, Y/dy, U_right, np.zeros((nx, ny))))
            ey = X[:nx+1, :]
            bz = X[nx+1:, :]

            #$ ey[0, :] = 1/dy*(ex[0, 1: ] + Ex[x1-1, y1+1:y2+1] - ex[0, :-1] - Ex[x1-1, y1:y2])  +  1/dx_f*(Ey[x1-1, y1:y2] + Ey[x1-2, y1:y2])  -  1/dt*(Bz[x1-1, y1:y2] - Bz_old[x1-1, y1:y2]) # UCHIE stitching left interface
            #$ ey[-1, :] = 1/dy*(ex[-1, 1: ] + Ex[x2+1, y1+1:y2+1] - ex[-1, :-1] - Ex[x2+1, y1:y2])  -  1/dx_f*(Ey[x2, y1:y2] + Ey[x2+1, y1:y2])  -  1/dt*(Bz[x2+1, y1:y2] - Bz_old[x2+1, y1:y2]) # UCHIE stitching right interface

            X = np.vstack((bz, ey))

            ### Field update in UCHIE region ex explicit ###
            ex[:, 1:-1] = ex[:, 1:-1] + dt/(mu0*eps0*dy) * (bz[:, 1:] - bz[:, :-1])

            #TODO something wrong in this
            A_1 = np.arange(nx+1)/nx # The interpolation matrix, see notes
            A_2 = A_1[::-1]
            A = np.vstack((A_2, A_1)).T


            ex[:, -1] = ex[:, -1]  +  dt/(mu0*eps0*dy) * (A @ Bz[x1:x2+1, y2] - bz[:, -1]) # Stitching upper interface @ Uchie
            ex[:, 0] = ex[:, 0]  -  mu0*dt/(eps0*dy) * (A @ Bz[x1:x2+1, y1-1] - bz[:, 0]) # Stitching down interface @ Uchie

            
            ### Update Ex and Ey in the Yee region ###
            Ex[1:-1, 1:-1] = Ex[1:-1, 1:-1] + dt/(dy*mu0*eps0) * (Bz[1:-1,1:] - Bz[1:-1,:-1])
            Ey = Ey - dt/(dx*mu0*eps0) * (Bz[1:,:] - Bz[:-1,:])
            Ey[x1-1,y1:y2] = Ey[x1-1,y1:y2] - dt/(dx*mu0*eps0) * bz[0, :] # stiching left interface
            Ey[x2,y1:y2] = Ey[x2,y1:y2] + dt/(dx*mu0*eps0) * bz[-1, :] # stiching right interface

            Ex[x1:x2+1, y1:y2+1] = 0 # Fields in the UCHIE region set zero to avoid double counting
            Ey[x1:x2, y1:y2] = 0 # Fields in the UCHIE region set zero to avoid double counting



            ### Save the data's ###
            data.append(copy.deepcopy(Bz.T))
            data_time.append(n*dt)


        return data_time, data



    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(data[0])#, vmin = -1e-13, vmax = 1e-13)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + str(t[i]))#"{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()



            
            
 
        



########## Fill in the parameters here ################
Nx = 100
Ny = 100
Nt = 400

dx = 0.25e-10 # m
dy = 0.25e-10 # ms
courant = 0.1 # !Courant number, for stability this should be smaller than 1
dt = courant * 1/(np.sqrt(1/dx**2 + 1/dy**2)*ct.c)

Ly = 50*dy
nx = 10 #@ Subgridding

x_sub = Nx//2 #@ The index where the subgridding should happen


#create the source
xs = Nx/4*dx
ys = Ny/2*dy
tc = dt*Nt/4
sigma = tc/10
source = Source(xs, ys, 1, tc, sigma)

test = Yee_UCHIE(Nx, Ny, Nt, dx, dy, dt, Ly, x_sub, nx, recorders=None)
M1_inv, M2, Ex, Ey, Bz, X, ex, ny, dx_f = test.initialize(Nx, Ny, Nt, dx, dy, dt, Ly, x_sub, nx)
data_time, data = test.calculate_fields(dx, dx_f, dy, dt, Ny, nx, ny, Ex, Ey, Bz, X, ex, M1_inv, M2, x_sub, Nt, source)
test.animate_field(data_time, data)