import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.animation as animation 
import pandas as pd
import matplotlib.patches as patch
import scipy.constants as ct


import QM_update as QM


eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass*0.15
q = -ct.elementary_charge
c0 = ct.speed_of_light 


Z0 = np.sqrt(mu0/eps0)


# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=20)

class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma

    def J(self, t):
        t = torch.tensor(t, device=device,dtype=torch.float64)
        return self.J0 * torch.exp(-(t - self.tc)**2 / (2 * self.sigma**2))
def genKy(Ny, Nx, pml_nl, m, dy):
    # Initialize Ky tensor on the device (GPU if available)
    Ky = torch.zeros((Nx + 1, Ny + 1), device=device,dtype=torch.float64)
    # Compute kmax using PyTorch operations
    kmax = -torch.log(torch.exp(torch.tensor(-16.0)).to(device)) * (m + 1) / (2 * np.sqrt(mu0 / eps0) * pml_nl * dy)
    
    # Generate the decay factors for the PML
    for iy in range(0, pml_nl):
        decay_factor = (kmax * ((pml_nl - 1 - iy) / pml_nl) ** m).to(device)
        Ky[:, iy] = decay_factor
        Ky[:, -iy - 1] = decay_factor
    
    return Ky

def genKx(Ny, Nx, pml_nl, m, dx):
    # Initialize Kx tensor on the device (GPU if available)
    Kx = torch.zeros((Nx + 1, Ny + 1), device=device,dtype=torch.float64)
    # Compute kmax using PyTorch operations
    kmax = -torch.log(torch.exp(torch.tensor(-16.0)).to(device)) * (m + 1) / (2 * np.sqrt(mu0 / eps0) * pml_nl * dx)
    
    # Generate the decay factors for the PML
    for ix in range(0, pml_nl):
        decay_factor = (kmax * ((pml_nl - 1 - ix) / pml_nl) ** m).to(device)
        Kx[ix, :] = decay_factor
        Kx[-ix - 1, :] = decay_factor
    
    return Kx

class Recorder:
    def __init__(self, x, y):
        self.x = torch.tensor(x, device=device,dtype=torch.float64)
        self.y = torch.tensor(y, device=device,dtype=torch.float64)
        self.data = []          # data of the field will be added in this list, each entry will be a tensor
        self.data_time = []     # data of the time will be added in this list

    # adding the measurement of the field to the data
    def save_data(self, field, t):
        # Ensure field data is a tensor and on the correct device
        if field.is_cuda:
            field = field.cpu()
        
        
        self.data.append(field.clone())  # appending a measurement to the list
        self.data_time.append(t)  # storing time as a tensor on the GPU


class Yee_UCHIE:
    def __init__(self, Nx, Ny, Nt, dx, dy, dt, Ly, n, N_sub, NYQM, x_sub1, x_sub2, eps, mu, source, QMscheme1, QMscheme2, QMxpos1, QMxpos2, recorders=[]):
        # Convert all constants to tensors and ensure they are on the correct device
        self.mu = torch.tensor(mu, device=device,dtype=torch.float64)
        self.eps = torch.tensor(eps, device=device,dtype=torch.float64)

        # Parameters of the simulation grid
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.dx = dx
        self.dx_f = self.dx / n
        self.dy = dy
        self.dt = dt
        self.Ly = Ly
        self.n = n
        self.N_sub = N_sub
        self.nx = n * N_sub
        self.ny = int(Ly / dy)  # Height of subgrid region
        self.NyQM = NYQM
        self.source = source
        self.xs = round(source.x / dx)
        self.ys = round(source.y / dy)

        self.QMscheme1 = QMscheme1
        self.QMscheme2 = QMscheme2

        self.QMxpos1 = QMxpos1
        self.QMxpos2 = QMxpos2

        self.recorders = recorders

        # Calculate position indices for subgrids
        self.x_sub1 = x_sub1 
        self.x_sub2 = x_sub2 
        self.x1 = self.x_sub1
        self.x2 = self.x1 + N_sub * dx
        self.x3 = self.x_sub2
        self.x4 = self.x3 + N_sub * dx
        self.y1 = (Ny - self.ny) // 2 * dy
        self.y2 = self.y1 + self.ny * dy

        # Initialize fields as zero tensors on GPU
        self.Ex = torch.zeros((Nx + 1, Ny + 1), device=device,dtype=torch.float64)
        self.Ey = torch.zeros((Nx, Ny), device=device,dtype=torch.float64)
        self.Bz = torch.zeros((Nx + 1, Ny), device=device,dtype=torch.float64)
        self.Bzx = torch.zeros((Nx + 1, Ny), device=device,dtype=torch.float64)
        self.Bzy = torch.zeros((Nx + 1, Ny), device=device,dtype=torch.float64)

        # Compute PML factors
        Ky = genKy(Ny, Nx, 1, 4, dy)
        Kx = genKx(Ny, Nx, 1, 4, dx)

        # Convert numpy arrays to tensors
        self.KxE = (2 * torch.full((Nx + 1, Ny + 1), eps, device=device,dtype=torch.float64) - Kx * dt) / (2 * torch.full((Nx + 1, Ny + 1), eps, device=device,dtype=torch.float64) + Kx * dt)
        self.KyE = (2 * torch.full((Nx + 1, Ny + 1), eps, device=device,dtype=torch.float64) - Ky * dt) / (2 * torch.full((Nx + 1, Ny + 1), eps, device=device,dtype=torch.float64) + Ky * dt)
        self.KxB = (2 * torch.full((Nx + 1, Ny + 1), mu, device=device,dtype=torch.float64) - mu * Kx * dt / eps) / (2 * torch.full((Nx + 1, Ny + 1), mu, device=device,dtype=torch.float64) + mu * Kx * dt / eps)
        self.KyB = (2 * torch.full((Nx + 1, Ny + 1), mu, device=device,dtype=torch.float64) - mu * Ky * dt / eps) / (2 * torch.full((Nx + 1, Ny + 1), mu, device=device,dtype=torch.float64) + mu * Ky * dt / eps)

        self.KxEB = (2 * dt) / ((2 * torch.full((Nx + 1, Ny + 1), eps, device=device,dtype=torch.float64) + Kx * dt) * dx * mu)
        self.KyEB = (2 * dt) / ((2 * torch.full((Nx + 1, Ny + 1), eps, device=device,dtype=torch.float64) + Ky * dt) * dy * mu)
        self.KxBE = (2 * mu * dt) / ((2 * torch.full((Nx + 1, Ny + 1), mu, device=device,dtype=torch.float64) + mu * Kx * dt / eps) * dx)
        self.KyBE = (2 * mu * dt) / ((2 * torch.full((Nx + 1, Ny + 1), mu, device=device,dtype=torch.float64) + mu * Ky * dt / eps) * dy)

        # Initializations for the UCHIE calculations
        self.X1 = torch.zeros((2 * self.nx + 2, self.ny), device=device,dtype=torch.float64)
        self.ex1 = torch.zeros((self.nx + 1, self.ny + 1), device=device,dtype=torch.float64)
        self.X2 = torch.zeros((2 * self.nx + 2, self.ny), device=device,dtype=torch.float64)
        self.ex2 = torch.zeros((self.nx + 1, self.ny + 1), device=device,dtype=torch.float64)
        self.ey1mid = torch.zeros((self.nx + 1, self.ny), device=device,dtype=torch.float64)
        self.ey2mid = torch.zeros((self.nx + 1, self.ny), device=device,dtype=torch.float64)

        self.data_yee = []
        self.data_uchie1 = []
        self.data_uchie2 = []
        self.data_time = []

       
        # Tensor-based matrix A_D
        A_D = torch.diag(torch.full((self.nx+1,), -1.0, device=device,dtype=torch.float64)) + \
              torch.diag(torch.ones(self.nx, device=device,dtype=torch.float64), 1)
        A_D = A_D[:-1, :]  # Slice to adjust shape as needed

        # Tensor-based matrix A_I
        A_I = torch.zeros((self.nx, self.nx + 1), device=device,dtype=torch.float64)
        indices = torch.arange(self.nx, device=device)
        A_I[indices, indices] = 1
        A_I[indices, indices + 1] = 1

        # Vector for M1_1 and M1_3 setup
        M1_1 = torch.zeros(2 * self.nx + 2, device=device,dtype=torch.float64)
        M1_1[0] = 1 / self.dx
        M1_1[self.nx + 1] = 1 / self.dt

        M1_2 = torch.cat((A_D / self.dx_f, A_I / self.dt), dim=1)

        M1_3 = torch.zeros(2 * self.nx + 2, device=device,dtype=torch.float64)
        M1_3[self.nx] = -1 / self.dx
        M1_3[-1] = 1 / self.dt

        M1_4 = torch.cat((self.eps * A_I / self.dt, A_D / (self.mu * self.dx_f)), dim=1)

        M1 = torch.cat((M1_1.unsqueeze(0), M1_2, M1_3.unsqueeze(0), M1_4), dim=0)
        self.M1_inv = torch.linalg.inv(M1)  # Compute matrix inverse using PyTorch

        # Setup for matrix M2 similar to M1
        M2_1 = torch.zeros(2 * self.nx + 2, device=device,dtype=torch.float64)
        M2_1[0] = -1 / self.dx
        M2_1[self.nx + 1] = 1 / self.dt

        M2_2 = torch.cat((-A_D / self.dx_f, A_I / self.dt), dim=1)

        M2_3 = torch.zeros(2 * self.nx + 2, device=device,dtype=torch.float64)
        M2_3[self.nx] = 1 / self.dx
        M2_3[-1] = 1 / self.dt

        M2_4 = torch.cat((self.eps / self.dt * A_I, -A_D / (self.mu * self.dx_f)), dim=1)

        self.M2 = torch.cat((M2_1.unsqueeze(0), M2_2, M2_3.unsqueeze(0), M2_4), dim=0)
        # Rest of the initialization code...

    def calculate_fields(self):
        # Ensure dimensions are integers
        
        x1, x2, x3, x4 = int(self.x1 / self.dx), int(self.x2 / self.dx), int(self.x3 / self.dx), int(self.x4 / self.dx)
        y1, y2 = int(self.y1 / self.dy), int(self.y2 / self.dy)
        for time_step in range(0, self.Nt):
            print(f"Time step: {time_step}")

            ### Field in Yee-region update ###
            Bz_old = self.Bz.clone()
            # Update Bz fields using PyTorch operations
            print((self.KyBE[1:-1, :-1] + self.KyBE[1:-1, 1:])[self.xs, self.ys])
            print((self.KyB[1:-1, :-1] + self.KyB[1:-1, 1:])[self.xs, self.ys])
            print("EX: \n")
            print((self.Ex[1:-1, 1:] )[self.xs, self.ys])
            print(self.Ex[1:-1, :-1][self.xs, self.ys])
            self.Bzy[1:-1, :] = ((self.KyB[1:-1, :-1] + self.KyB[1:-1, 1:]) / 2 * self.Bzy[1:-1, :] + \
                                (self.KyBE[1:-1, :-1] + self.KyBE[1:-1, 1:]) / 2 * (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1])).clone()
            self.Bzx[1:-1, :] = ((self.KxB[1:-1, :-1] + self.KxB[1:-1, 1:]) / 2 * self.Bzx[1:-1, :] - \
                                (self.KxEB[1:-1, :-1] + self.KxEB[1:-1, 1:]) / 2 * (self.Ey[1:, :] - self.Ey[:-1, :])).clone()
            # Add source terms
            source_term = self.dt * self.source.J(time_step * self.dt) / 2
            self.Bzy[self.xs, self.ys] -= source_term
            self.Bzx[self.xs, self.ys] -= source_term

            # Stitching B fields
            # self.Bzx = self.stitching_B(x1, x2, y1, y2, self.Bzx, self.ex1, self.n, self.dt, self.dy)
            # self.Bzy = self.stitching_B(x1, x2, y1, y2, self.Bzy, self.ex1, self.n, self.dt, self.dy)
            # self.Bzx = self.stitching_B(x3, x4, y1, y2, self.Bzx, self.ex2, self.n, self.dt, self.dy)
            # self.Bzy = self.stitching_B(x3, x4, y1, y2, self.Bzy, self.ex2, self.n, self.dt, self.dy)
            self.Bz = (self.Bzx + self.Bzy).clone()
            print(source_term)
            print(self.Bz[self.xs, self.ys])
            print("BZ: \n")
            print((self.Bz[1:-1, 1:] )[self.xs, self.ys])
            print(self.Bz[1:-1, :-1][self.xs, self.ys])
            # Update QM    schemes
            slice_idx = int((self.ny - self.NyQM) / 2)
            self.QMscheme1.update(self.ey1mid[self.QMxpos1, slice_idx:-slice_idx], time_step)
            self.QMscheme2.update(self.ey2mid[self.QMxpos2, slice_idx:-slice_idx], time_step)

            
            ### Field update in UCHIE region updated, bz and ey with implicit ###
            self.X1, self.ex1, self.ey1, self.bz1, self.ey1mid = self.implicit_update(self.X1, self.ex1, self.Ex, self.Ey, self.Bz, Bz_old, x1, x2, y1, y2, self.nx, self.ny, self.n, self.N_sub, self.dx_f, self.dx, self.dy, self.dt, self.M1_inv, self.M2, self.eps, self.mu,QMscheme1.Jmid, QMxpos1)
            self.X2, self.ex2, self.ey2, self.bz2 , self.ey2mid= self.implicit_update(self.X2, self.ex2, self.Ex, self.Ey, self.Bz, Bz_old, x3, x4, y1, y2, self.nx, self.ny, self.n, self.N_sub, self.dx_f, self.dx, self.dy, self.dt, self.M1_inv, self.M2, self.eps, self.mu, QMscheme2.Jmid, QMxpos2)



            ### Update Ex and Ey in the Yee region ###
            print(self.KyEB[1:-1, 1:-1][self.xs, self.ys])  
            self.Ex[1:-1, 1:-1] = (self.KyE[1:-1, 1:-1] * self.Ex[1:-1, 1:-1] + \
                                   self.KyEB[1:-1, 1:-1] * (self.Bz[1:-1, 1:] - self.Bz[1:-1, :-1])).clone()
            self.Ey = ((self.KxE[:-1, :-1] + self.KxE[1:, :-1] + self.KxE[:-1, 1:] + self.KxE[1:, 1:]) / 4 * self.Ey - \
                      (self.KxEB[:-1, :-1] + self.KxEB[1:, :-1] + self.KxEB[:-1, 1:] + self.KxEB[1:, 1:]) / 4 * (self.Bz[1:, :] - self.Bz[:-1, :])).clone()

            # self.Ex, self.Ey = self.stitching_E(x1, x2, y1, y2, self.Ex, self.Ey, self.bz1, self.dx, self.dt, self.mu, self.eps)
            # self.Ex, self.Ey = self.stitching_E(x3, x4, y1, y2, self.Ex, self.Ey, self.bz2, self.dx, self.dt, self.mu, self.eps)
            print(self.Ex[self.xs, self.ys])
            print(self.Ey[self.xs, self.ys])
            if time_step%(self.Nt/1000)==0:
                #print(n)
                # self.data_yee.append(copy.deepcopy(self.Bz.T))
                # self.data_uchie1.append(copy.deepcopy(self.bz1.T))
                # self.data_uchie2.append(copy.deepcopy(self.bz2.T))
                self.data_yee.append((self.Bz.T).cpu())
                self.data_uchie1.append((self.bz1.T).cpu())
                self.data_uchie2.append((self.bz2.T).cpu())
                self.data_time.append((time_step*self.dt))
    def create_interpolation_matrix(self, n, nx, N_sub):
        # Initialize the matrix A_pol on the GPU with zeros
        A_pol = torch.zeros((nx + 1, N_sub + 1), device=device,dtype=torch.float64)

        # Generate the interpolation matrix using PyTorch operations
        for i in range(N_sub):
            # Create the interpolation vectors
            A_1 = torch.arange(n + 1, device=device,dtype=torch.float64) / n
            A_2 = A_1.flip(dims=[0])  # Reversing the tensor
            
            # Stack the vectors vertically and transpose to form the matrix A
            A = torch.stack((A_2, A_1), dim=0).T
            A = A[:-1, :]  # Adjust the shape by removing the last row
            # Place the small matrix A into the larger matrix A_pol
            A_pol[i * n:(i + 1) * n, i:i + 2] = A

        # Set the last element
        A_pol[-1, -1] = 1
        return A_pol

    def implicit_update(self, X, ex, Ex, Ey, Bz, Bz_old, x1, x2, y1, y2, nx, ny, n, N_sub, dx_f, dx, dy, dt, M1_inv, M2, eps, mu, JQM, QMxpos):
        # Calculate Y using tensor operations
        Y = ex[:-1, 1:] + ex[1:, 1:] - ex[:-1, :-1] - ex[1:, :-1]

        # Update specific region based on quantum mechanics input
        slice = int((self.ny - self.NyQM) / 2)
        Y[QMxpos, slice:-slice] -= 2 * (1 / Z0) * 0#JQM
        
        # Save the old ey field values
        eyold = X[:nx + 1, :]

        # Calculate U_left and U_right using tensor operations
        U_left = (1 / dy) * (ex[0, 1:] + Ex[x1 - 1, y1 + 1:y2 + 1] - ex[0, :-1] - Ex[x1 - 1, y1:y2]) + \
                 (1 / dx) * (Ey[x1 - 1, y1:y2] + Ey[x1 - 2, y1:y2]) - \
                 (1 / dt) * (Bz[x1 - 1, y1:y2] - Bz_old[x1 - 1, y1:y2])
        
        U_right = (1 / dy) * (ex[-1, 1:] + Ex[x2 + 1, y1 + 1:y2 + 1] - ex[-1, :-1] - Ex[x2 + 1, y1:y2]) - \
                  (1 / dx) * (Ey[x2, y1:y2] + Ey[x2 + 1, y1:y2]) - \
                  (1 / dt) * (Bz[x2 + 1, y1:y2] - Bz_old[x2 + 1, y1:y2])

        # Update X using matrix operations in PyTorch
        U_vector = torch.cat((U_left.unsqueeze(0), Y / dy, U_right.unsqueeze(0), torch.zeros((nx, ny), device=device)))
        X = torch.matmul(M1_inv, torch.matmul(M2, X) + U_vector)

        # Split X into ey and bz
        ey = X[:nx + 1, :]
        bz = X[nx + 1:, :]

        # Update ex fields in the UCHIE region
        ex[:, 1:-1] += dt / (mu * eps * dy) * (bz[:, 1:] - bz[:, :-1])

        # Create interpolation matrix and use it for stitching
        A_pol = self.create_interpolation_matrix(n, nx, N_sub)
        ex[:, -1] += dt / (mu * eps * dy) * torch.matmul(A_pol, Bz[x1:x2 + 1, y2]) - bz[:, -1]
        ex[:, 0] -= dt / (mu * eps * dy) * torch.matmul(A_pol, Bz[x1:x2 + 1, y1 - 1]) - bz[:, 0]

        # Calculate eymid for return
        eymid = 0.5 * (eyold + ey)

        return X, ex, ey, bz, eymid

    def stitching_B(self, x1, x2, y1, y2, Bz, ex, n, dt, dy):
        # Set the B fields in the UCHIE region to zero to prevent double counting
        Bz[x1:x2+1, y1:y2] = 0
        # Update the boundary conditions using tensor slicing and operations
        Bz[x1:x2+1, y2] -= (dt / dy) * ex[::n, -1] / 2  # Stitching upper interface
        Bz[x1:x2+1, y1-1] += (dt / dy) * ex[::n, 0] / 2  # Stitching lower interface
        return Bz

    def stitching_E(self, x1, x2, y1, y2, Ex, Ey, bz, dx, dt, mu, eps):
        # Adjust the electric fields at the boundaries of the UCHIE region
        Ey[x1-1, y1:y2] -= dt / (dx * mu * eps) * bz[0, :]  # Stitching left interface
        Ey[x2, y1:y2] += dt / (dx * mu * eps) * bz[-1, :]  # Stitching right interface

        # Zero out the electric fields in the UCHIE region to avoid double counting
        Ex[x1:x2+1, y1:y2+1] = 0
        Ey[x1:x2, y1:y2] = 0
        
        return Ex, Ey
    
    def animate_field(self):

        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        ax.set_xlim(0, (self.Nx+1)*self.dx)
        ax.set_ylim(0, (self.Ny+1)*self.dy)

        label = "Field"

        xs = self.source.x
        ys = self.source.y
        
        # v = source.J0*dt * 0.05
        v = np.max(self.data_yee)*0.5
        # vmax = np.percentile(electric_field, 95)
        # v = source.J0*self.dx*self.dy/2*1000/Nt
        #v = 1e2
        

        ax.plot(xs, ys+0.5*self.dy, color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(self.data_yee[0], vmin = -v, vmax = v, origin='lower', extent = [0, (self.Nx+1)*self.dx, self.dy/2, self.Ny*self.dy])
        ax.set_title("T = 0")

        subgrid1 = [self.x1, self.x2, self.y1, self.y2]
        scax1 = ax.imshow(self.data_uchie1[0], vmin=-v, vmax=v, origin='lower', extent=subgrid1)
        rect1=patch.Rectangle((self.x1, self.y1),self.x2-self.x1, self.y2-self.y1, alpha = 0.05, facecolor="grey", edgecolor="black")
        ymin = (int(1/2*(self.Ny-self.ny))+int(1/2*(self.ny-self.NyQM)))*self.dy
        ymax = (self.Ny- int(1/2*(self.Ny-self.ny))-int(1/2*(self.ny-self.NyQM)))*self.dy
        ax.vlines(self.x1+self.QMxpos1//n*self.dx, ymin=ymin, ymax = ymax, color='red', linewidth=1)

        ax.add_patch(rect1)

        subgrid2 = [self.x3, self.x4, self.y1, self.y2]
        scax2 = ax.imshow(self.data_uchie2[0], vmin=-v, vmax=v, origin='lower', extent=subgrid2)
        rect2=patch.Rectangle((self.x3, self.y1),self.x4-self.x3, self.y2-self.y1, alpha = 0.05, facecolor="grey", edgecolor="black")
        ymin = (int(1/2*(self.Ny-self.ny))+int(1/2*(self.ny-self.NyQM)))*self.dy
        ymax = (self.Ny- int(1/2*(self.Ny-self.ny))-int(1/2*(self.ny-self.NyQM)))*self.dy
        ax.vlines(self.x3+self.QMxpos2//self.n*self.dx, ymin=ymin, ymax = ymax, color='red', linewidth=1)
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
Nx = 20
Ny = 20
Nt = 3

dx = 0.25e-10 # m
dy = 0.25e-10 # ms
courant = 0.9 # !Courant number, for stability this should be smaller than 1
dt = courant * 1/(np.sqrt(1/dx**2 + 1/dy**2)*ct.c)

Ly = 3/5*Ny*dy
n = 2 #@ numbers of Subgridding in one grid
N_sub = 2 #@ How much grid you want to take to subgrid

x_sub1 = Nx//3*dx #@ The locationwhere the first subgridding should happen
x_sub2 = 2*Nx//3*dx #@ The locationwhere the first subgridding should happen

QMxpos1 = n*N_sub//2
QMxpos2 = n*N_sub//2


NyQM = int(2*Ny/5)

#create the source
xs = 1/4*Nx*dx
ys = Ny/2*dy
tc = dt*Nt/4
sigma = tc/10
J0 = 1e2/dx/dy
source = Source(xs, ys, J0, tc, sigma)

N = 10e7 #particles/m2
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

QMscheme1.expvalues('energy')
QMscheme2.expvalues('energy')

