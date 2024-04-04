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
    

    def initialize(self, Nx, Ny):
        # Ey = np.zeros(Nx+1, Ny)
        # Hz = Ey

        X = np.zeros((2*Nx+2, Ny))
        Ex = np.zeros((Nx+1, Ny+1))

        A_D = np.diag(-1 * np.ones(Nx+1), 0) + np.diag(np.ones(Nx), 1)

        A_I = np.zeros((Nx+1, Nx+1))
        np.fill_diagonal(A_I, 1)
        np.fill_diagonal(A_I[:,1:], 1)

        return X, Ex, A_D, A_I



    ### Update ###
    def explicit(self, Ex, Hz, dy, dt, eps):
        Ex = Ex + dt/(eps*dy) * (Hz[:, 1:] - Hz[:, :-1])

    # ! It can be that the matrix M1 isn't constructed correctly, since the original
    # ! matrix in the paper is undertermined, artificial zeros was added here. In paper
    # ! interface condition was applied
    def implicit(self, n, X, Ex, dx, dy, dt, Nx, Ny, eps, mu, A_D, A_I, source):
        Y = Ex[:-1, 1:] + Ex[1:, 1:] - Ex[:-1, :-1] - Ex[1:, :-1]

        # We will create a bigger matrix which we need to inverse
        M1_top = np.hstack((1/dx*A_D, 1/dt*A_I))
        M1_bot = np.hstack((eps/dt*A_I, -1/(mu*dx)*A_D))
        M1 = np.vstack((M1_top, M1_bot))

        M2_top = np.hstack((1/dx*A_D, 1/dt*A_I))
        M2_bot = np.hstack((eps/dt*A_I, -1/(mu*dx)*A_D))
        M2 = np.vstack((M2_top, M2_bot))

        S = np.zeros((2*Nx+2, Ny))
        S[xs, ys] = source.J(n*dt)

        M1_inv = np.linalg.inv(M1)
        X = M1_inv @ M2 @ X + M1_inv @ np.vstack((Y, np.zeros((Nx+2, Ny))))/dy - M1_inv @ S

        return X
        
    
    def calc_field(self, X, Ex):
        pass


eps = 8.854 * 10**(-12)
mu = 4*np.pi * 10**(-7)

dx = 0.1 # m
dy = 0.1 # m
c = 299792458 # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c

Nx = 10
Ny = 10

xs = 3
ys = 3


source = Source(3, 3, 10, 5e-9, 1e-9)

test = UCHIE()
X, Ex, A_D, A_I = test.initialize(Nx, Ny)
print(test.implicit(1, X, Ex, dx, dy, dt, Nx, Ny, eps, mu, A_D, A_I, source))

print('next step')
print(test.implicit(2, X, Ex, dx, dy, dt, Nx, Ny, eps, mu, A_D, A_I, source))
print('next step')
print(test.implicit(3, X, Ex, dx, dy, dt, Nx, Ny, eps, mu, A_D, A_I, source))