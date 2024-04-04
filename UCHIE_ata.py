import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 



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
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))



#### UCHIE ####
class UCHIE:
    def __init__(self):
        pass
        # self.dx = dx
        # self.dy = dy

        # self.Lx = Lx
        # self.Ly = Ly
    

    def initialize(self, Nx, Ny, eps, mu):
        # Ey = np.zeros(Nx+1, Ny)
        # Hz = Ey

        X = np.zeros((2*Nx+2, Ny))
        Ex = np.zeros((Nx+1, Ny+1))

        A_D = np.diag(-1 * np.ones(Nx+1), 0) + np.diag(np.ones(Nx), 1)

        A_I = np.zeros((Nx+1, Nx+1))
        np.fill_diagonal(A_I, 1)
        np.fill_diagonal(A_I[:,1:], 1)

        # We will create a bigger matrix which we need to inverse
        M1_top = np.hstack((1/dx*A_D, 1/dt*A_I))
        M1_bot = np.hstack((eps/dt*A_I, -1/(mu*dx)*A_D))
        M1 = np.vstack((M1_top, M1_bot))
        M1_inv = np.linalg.inv(M1)

        M2_top = np.hstack((1/dx*A_D, 1/dt*A_I))
        M2_bot = np.hstack((eps/dt*A_I, -1/(mu*dx)*A_D))
        M2 = np.vstack((M2_top, M2_bot))

        return X, Ex, M1_inv, M2



    ### Update ###
    def explicit(self, Ex, Hz, dy, dt, eps):
        Ex[:, 1:-1] = Ex[:, 1:-1] + dt/(eps*dy) * (Hz[:, 1:] - Hz[:, :-1])
        return Ex

    # ! It can be that the matrix M1 isn't constructed correctly, since the original
    # ! matrix in the paper is undertermined, artificial zeros was added here. In paper
    # ! interface condition was applied
    def implicit(self, n, X, Ex, dx, dy, dt, Nx, Ny, M1_inv, M2, source):
        Y = Ex[:-1, 1:] + Ex[1:, 1:] - Ex[:-1, :-1] - Ex[1:, :-1]

        S = np.zeros((2*Nx+2, Ny))
        S[xs, ys] = source.J(n*dt)

        X = M1_inv @ M2 @ X + M1_inv @ np.vstack((Y, np.zeros((Nx+2, Ny))))/dy - M1_inv @ S

        return X
        
    
    def calc_field(self, dx, dy, dt, Nx, Ny, Nt, eps, mu, source):
        X, Ex, M1_inv, M2 = self.initialize(Nx, Ny, eps, mu)
        data_time = []
        data = []

        for n in range(1, Nt):
            X = self.implicit(n, X, Ex, dx, dy, dt, Nx, Ny, M1_inv, M2, source)
            Ex = self.explicit(Ex, X[Nx+1:,:], dy, dt, eps,)
            data_time.append(dt*n)
            data.append(Ex)
        
        return data_time, data

    # TODO animate_field function doesn't work at all, I don't see the fields
    def animate_field(self, t, data, source):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [m]")
        ax.set_ylabel("y-axis [m]")
        ax.set_xlim(0, Nx*dx)
        ax.set_ylim(0, Ny*dy)

        label = "P-field"
        
        ax.plot(source.x, source.y) # plot the source

        cax = ax.imshow(data[0])
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)))
        plt.show()



##########################################################

        


eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

dx = 0.1 # m
dy = 0.1 # m
c = 299792458 # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c

Nx = 10
Ny = 10
Nt = 10

xs = 3
ys = 3


source = Source(3, 3, 10, 5e-9, 1e-9)

test = UCHIE()
data_time, data = test.calc_field(dx, dy, dt, Nx, Ny, Nt, eps0, mu0, source)
test.animate_field(data_time, data, source)