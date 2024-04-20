import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
import pandas as pd
import scipy as sp



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
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))#*np.cos(2e10*t)



#### UCHIE ####
class UCHIE:
    def __init__(self):
        pass
        # self.dx = dx
        # self.dy = dy

        # self.Lx = Lx
        # self.Ly = Ly
    

    def initialize(self, Nx, Ny, eps, mu):

        X = np.zeros((2*Nx+2, Ny)) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
        Ex = np.zeros((Nx+1, Ny+1))
        

        A_D = np.diag(-1 * np.ones(Nx+1), 0) + np.diag(np.ones(Nx), 1)
        A_D[Nx, 0] = 1
        #print(A_D.shape)

        A_I = np.diag(1 * np.ones(Nx+1), 0) + np.diag(np.ones(Nx), 1)
        A_I[Nx, 0] = 1

        # We will create a bigger matrix which we need to inverse
        M1_top = np.hstack((1/dx*A_D, 1/dt*A_I))
        M1_bot = np.hstack((eps/dt*A_I, 1/(mu*dx)*A_D))
        
        M1 = np.vstack((M1_top, M1_bot))
        # M1[0, :] = np.zeros(2*Nx+2)
        # M1[0, 0] = 1
        # M1[Nx + 1, :] = np.zeros( 2*Nx+2)
        # M1[Nx + 1, Nx + 1] = 1
        M1_inv = sp.linalg.inv(M1)

        M2_top = np.hstack((-1/dx*A_D, 1/dt*A_I))
        M2_bot = np.hstack((eps/dt*A_I, -1/(mu*dx)*A_D))
        
        M2 = np.vstack((M2_top,  M2_bot))

        # M2[0, :] = np.zeros(2*Nx+2)
        
        # M2[Nx + 1, :] = np.zeros( 2*Nx+2)
        return X, Ex, M1_inv, M2



    ### Update ###
    def explicit(self, n, Ex, Bz, dy, dt, eps, mu):

        # S = np.zeros((Nx-1, Ny-1))
        # S[ int(source.x/dx), int(source.y/dy)] = source.J(n*dt)*dt
        Ex[1:-1, 1:-1] = Ex[1:-1, 1:-1] + dt/(eps*dy*mu) * (Bz[1:-1,1: ] - Bz[1:-1,:-1 ]) 


        return Ex


    def implicit(self, n, X, Ex, dx, dy, dt, Nx, Ny, M1_inv, M2, source):
        
        Y = Ex[:-1, 1:] + Ex[1:, 1:] - Ex[:-1, :-1] - Ex[1:, :-1]

        S = np.zeros((2*Nx+2, Ny))
        S[  int(source.x/dx), int(source.y/dy)] = source.J(n*dt)
        #S[Nx+ int(source.x/dx), int(source.y/dy)] = source.J(n*dt)
    
        #X = M1_inv@M2@X + M1_inv@np.vstack((Y, S))/dy #+ M1_inv@S
        X = M1_inv@M2@X + M1_inv@np.vstack((Y, np.zeros((Nx+2, Ny))))/dy +M1_inv@S
        

        return X
        
    
    def calc_field(self, dx, dy, dt, Nx, Ny, Nt, eps, mu, source):
        X, Ex, M1_inv, M2 = self.initialize(Nx, Ny, eps, mu)
        data_time = []
        data = []

        for n in range(0, Nt):
            X= self.implicit(n, X ,Ex, dx, dy, dt, Nx, Ny, M1_inv, M2, source)
            Ex = self.explicit(n, Ex, X[Nx+1:,:], dy, dt, eps, mu)
            data_time.append(dt*n)
            data.append(copy.deepcopy((X[Nx+1:,:].T)))
            
        
        return data_time, data

    # TODO animate_field function doesn't work, I don't see the field propagating
    def animate_field(self, t, data, source, dx, dy, Nx, Ny):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(data[0])
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()



##########################################################

        


eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

dx = 0.008 # m
dy = 0.01 # ms
c = 299792458 # m/s
Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c

Nx = 300
Ny = 300
Nt = 500

# dx = 1 # m
# dy = 2 # ms
# c = 299792458 # m/s
# Sy = 0.1 # !Courant number, for stability this should be smaller than 1
# dt = Sy*dy/c

# Nx = 10
# Ny = 5
# Nt = 100


xs = Nx*dx/2
ys = Ny*dy/2

tc = dt*Nt/2
print(tc)
sigma = tc/6

source = Source(xs, ys, 1, tc, sigma)

test = UCHIE()
data_time, data = test.calc_field(dx, dy, dt, Nx, Ny, Nt, eps0, mu0, source)
test.animate_field(data_time, data, source, dx, dy, Nx, Ny)
