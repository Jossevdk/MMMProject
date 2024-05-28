import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation

# Check if GPU is available and set it as the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ElectricField:
    def __init__(self, field_type, dt, amplitude=1.0):
        self.field_type = field_type
        self.amplitude = amplitude
        self.dt = dt

    def generate(self, t, **kwargs):
        if self.field_type == 'gaussian':
            return self._gaussian(t, **kwargs)
        elif self.field_type == 'sinusoidal':
            return self._sinusoidal(t, **kwargs)
        
        #add a third case where these is coupling with the EM part
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")

    def _gaussian(self, t, t0=0, sigma=1):
        t0 = 20000*self.dt
        sigma = t0/5
        return self.amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    def _sinusoidal(self, t, omega=1):
        t0= 1000*self.dt
        #add damping function
        return self.amplitude * np.sin(omega * t)*2/np.pi* np.arctan(t/t0)

    # Initialization remains the same; calculations will be adjusted later.

class Potential:
    def __init__(self, m, omega, Ny, dy):
        self.m = torch.tensor(m, device=device)
        self.omega = torch.tensor(omega, device=device)
        self.Ny = Ny
        self.dy = torch.tensor(dy, device=device)
        # Compute potential using torch operations and ensure it's on GPU
        self.V = 0.5 * self.m * self.omega**2 * torch.linspace(-self.Ny//2 * self.dy, self.Ny//2 * self.dy, self.Ny, device=device)**2

class QM:
    def __init__(self, order, Ny, dy, dt, hbar, m, q, alpha, potential, omega, N):
        self.Ny = Ny
        self.dy = dy
        self.dt = dt
        self.hbar = hbar
        self.m = m
        self.q = q
        self.alpha=alpha
        self.result = None
        self.order = order
        self.potential = potential
        self.omega =omega
        self.N=N
        
        # Initialize other tensors on GPU
        self.r = torch.linspace(-self.Ny/2 * self.dy, self.Ny/2 * self.dy, self.Ny, device=device)
        self.PsiRe = (self.m * self.omega / (torch.pi * self.hbar))**0.25 * torch.exp(-self.m * self.omega / (2 * self.hbar) * (self.r - self.alpha * np.sqrt(2 * self.hbar / (self.m * self.omega)))**2)
        self.PsiIm = torch.zeros(self.Ny, device=device)
        self.Jmid = torch.zeros(self.Ny, device=device)
        self.J = torch.zeros(self.Ny, device=device)
        # Ensure other vectors are initialized on GPU as needed

        self.data_prob= []
        self.data_time = []
        self.data_mom=[]
        self.data_energy= []
        self.beam_energy=[]
        self.data_current = []
        self.data_position= []

    def diff(self, psi):
        if self.order == 'second':
            psi = (torch.roll(psi, 1, 0) - 2 * psi + torch.roll(psi, -1, 0)) / self.dy**2
        elif self.order == 'fourth':
            psi = (-torch.roll(psi, 2, 0) + 16 * torch.roll(psi, 1, 0) - 30 * psi + 16 * torch.roll(psi, -1, 0) - torch.roll(psi, -2, 0)) / (12 * self.dy**2)
        psi[0] = psi[-1] = psi[1] = psi[-2] = 0
        return psi

    def update(self, efield, n):
        # Generate electric field at the current time, converted to the same device as other tensors
        #E = efield.generate(n * self.dt).to(device) * torch.ones(self.Ny, device=device)

        # Calculate the potential array (if not constant)
        V = self.potential.V.to(device)
        efield = efield.to(device)
        # Update PsiRe
        # Calculate the kinetic and potential term contributions for PsiRe
        kinetic_term_Re = self.hbar * self.dt / (2 * self.m) * self.diff(self.PsiIm)
        potential_term_Re = self.dt / self.hbar * (self.q * self.r * efield - V) * self.PsiIm
        self.PsiRe -= kinetic_term_Re + potential_term_Re

        # Enforce boundary conditions for PsiRe
        self.PsiRe[0] = 0
        self.PsiRe[-1] = 0

        # Update PsiIm
        # Calculate the kinetic and potential term contributions for PsiIm
        PsiImo = self.PsiIm
        kinetic_term_Im = self.hbar * self.dt / (2 * self.m) * self.diff(self.PsiRe)
        potential_term_Im = self.dt / self.hbar * (self.q * self.r * efield - V) * self.PsiRe
        self.PsiIm += kinetic_term_Im + potential_term_Im

        # Enforce boundary conditions for PsiIm
        self.PsiIm[0] = 0
        self.PsiIm[-1] = 0

        # Calculate the current density J based on the midpoint rule for PsiRe and PsiIm
        PsiImhalf = (PsiImo + self.PsiIm)/2
        J_old= self.J
        self.J = self.q * self.N * self.hbar / (self.m * self.dy) * (self.PsiRe * torch.roll(PsiImhalf, -1, 0) - torch.roll(self.PsiRe, -1, 0) * PsiImhalf)

        # Enforce boundary conditions for current density J
        self.J[0] = 0
        self.J[-1] = 0
        self.Jmid = (self.J+J_old)/2
        # Optionally compute additional properties like probability density or momentum as needed
        prob = self.PsiRe**2 + PsiImhalf**2
        self.data_prob.append(prob.cpu())  # Store CPU copy for later visualization

        # Keep track of other quantities like total energy or beam energy if required
        Psi = self.PsiRe + 1j * PsiImhalf
        momentum = torch.conj(Psi) * -1j * self.hbar / (2 * self.dy) * (torch.roll(Psi, -1, 0) - torch.roll(Psi, 1, 0))
        energy = torch.sum(torch.conj(Psi) * (-self.hbar**2 / (2 * self.m) * self.diff(Psi) + V * Psi))
        beam_energy = torch.sum(torch.conj(Psi) * (-self.q * self.r * efield * Psi))

        # Append computed values to their respective lists
        self.data_mom.append(torch.sum(momentum).cpu().item())  # Convert to CPU for storage and visualization
        self.data_energy.append(energy.cpu().item())
        self.beam_energy.append(beam_energy.cpu().item())
        self.data_current.append(torch.sum(self.J).cpu().item())
        self.data_position.append(torch.sum(prob * self.r * self.dy).cpu().item())
    def expvalues(self, exp_type):
        # Select the data type based on input type and convert any necessary data from GPU to CPU
        if exp_type == 'position':
            # Compute expectation values of position
            exp_vals = [torch.sum(prob * self.r * self.dy).cpu().numpy() for prob in self.data_prob]
            plt.plot(self.data_time, exp_vals)
            plt.title('Expectation of Position')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.show()

        elif exp_type == 'momentum':
            # Convert momentum data directly for plotting
            plt.plot(self.data_time, self.data_mom)
            plt.title('Expectation of Momentum')
            plt.xlabel('Time (s)')
            plt.ylabel('Momentum (kg*m/s)')
            plt.show()

        elif exp_type == 'energy':
            # Plot the total energy over time
            plt.plot(self.data_time, self.data_energy, label='Total Energy')
            plt.plot(self.data_time, self.beam_energy, label='Beam Energy')
            plt.title('Energy Expectation Values')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (Joules)')
            plt.legend()
            plt.show()

        elif exp_type == 'current':
            # Plot the quantum current over time
            plt.plot(self.data_time, self.data_current)
            plt.title('Quantum Current')
            plt.xlabel('Time (s)')
            plt.ylabel('Current (A)')
            plt.show()

        else:
            raise ValueError(f"Unknown expectation type: {exp_type}")


    def animate(self):
        # Ensure Matplotlib is set up to run in the required environment (e.g., Jupyter notebook using '%matplotlib notebook')
        fig, ax = plt.subplots()
        
        # Set up the plot limits
        ax.set_xlim(0, len(self.data_prob[0]))
        ax.set_ylim(0, torch.max(torch.stack(self.data_prob)).cpu().item() + 0.1 * torch.max(torch.stack(self.data_prob)).cpu().item())

        # Initial empty plot
        line, = ax.plot([], [], lw=2)

        # Initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,

        # Animation update function, called sequentially
        def update(frame):
            y_data = self.data_prob[frame].cpu().numpy()  # Transfer tensor to CPU and convert to numpy for plotting
            x_data = torch.arange(len(y_data)).cpu().numpy()  # Corresponding x data
            line.set_data(x_data, y_data)
            return line,

        # Call the animator, blit=True means only re-draw the parts that have changed
        anim = FuncAnimation(fig, update, frames=len(self.data_time), init_func=init, blit=True, repeat=False)

        plt.show()
        return anim
