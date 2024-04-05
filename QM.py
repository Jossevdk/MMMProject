import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct

#For the QM part, we require a simple 1D FDTD scheme


#strategy: electric field from EM part is source in the interaction Hamiltionian. 
#Output is a quantum current which serves as a source for EM part


### Electric field ###
class ElectricField:
    def __init__(self, field_type, amplitude=1.0):
        self.field_type = field_type
        self.amplitude = amplitude

    def generate(self, t, **kwargs):
        if self.field_type == 'gaussian':
            return self._gaussian(t, **kwargs)
        elif self.field_type == 'sinusoidal':
            return self._sinusoidal(t, **kwargs)
        
        #add a third case where these is coupling with the EM part
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")

    def _gaussian(self, t, t0=0, sigma=1):
        return self.amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    def _sinusoidal(self, t, frequency=1):
        #add damping function
        return self.amplitude * np.sin(omega * t)



### Potential ###
class Potential:
    def __init__(self, m, omega,Ny,dy):
        self.m = m
        self.omega = omega
        self.Ny = Ny
        self.dy = dy
        
        
    #This will call the function depending on which type of source you have    
    def V(self):
        V = 0.5*self.m*self.omega**2* (np.linspace(-self.Ny//2*self.dy, self.Ny//2*self.dy,Ny))**2
        return V
    

#### QM ####
class QM:
    def __init__(self):
        pass
    def initialize(self, dy, Ny,m,omega, hbar,alpha):
        #coherent state at y=0 for electron
        #PsiRe = np.zeros(Ny)
        
        r = np.linspace(-Ny/2*dy, Ny/2*dy,Ny)
        print(r)
        PsiRe = (m*omega/(np.pi*hbar))**(1/4)*np.exp(-m*omega/(2*hbar)*(r-alpha*np.sqrt(2*hbar/(m*omega))*np.ones(Ny))**2)
        # plt.plot(r,PsiRe)
        # plt.show()
        #print(PsiRe)
        PsiIm = np.zeros(Ny)
        return PsiRe, PsiIm, r

    def second_order(self,psi,dy):
        psi= (np.roll(psi,1) -2*psi + np.roll(psi,-1))/dy**2
        psi[0] = 0
        psi[-1] = 0
        return psi

    ### Update ###
    def update(self, PsiRe, PsiIm, dy, dt, hbar, m, q, r, potential, efield, n):
        E = efield.generate(n*dt)*np.ones(Ny)
        #PsiRe = PsiRe -hbar*dt/(2*m)*self.second_order(PsiIm,dy) - dt/hbar*(q*r*E-potential.V())*PsiIm
        PsiRe = PsiRe -hbar*dt/(2*m)*self.second_order(PsiIm,dy) - dt/hbar*(-potential.V())*PsiIm
        PsiRe[0] = 0
        PsiRe[-1] = 0
        E = efield.generate(n+1/2*dt)*np.ones(Ny)
        #PsiIm = PsiIm +hbar*dt/(2*m)*self.second_order(PsiRe,dy) + dt/hbar*(q*r*E-potential.V())*PsiRe
        PsiIm = PsiIm +hbar*dt/(2*m)*self.second_order(PsiRe,dy) + dt/hbar*(-potential.V())*PsiRe
        PsiIm[0] = 0
        PsiIm[-1] = 0
        #Ex[:, 1:-1] = Ex[:, 1:-1] + dt/(eps*dy) * (Hz[:, 1:] - Hz[:, :-1])
        return PsiRe, PsiIm

    
        
    
    def calc_wave(self, dy, dt, Ny, Nt,  hbar, m ,q ,potential, efield,alpha):
        PsiRe,PsiIm ,r = self.initialize(dy, Ny,m,omega, hbar,alpha)
        data_time = []
        dataRe = []
        dataIm = []
        dataprob = []

        for n in range(1, Nt):
            PsiRe, PsiIm = self.update(PsiRe, PsiIm, dy, dt, hbar, m, q, r, potential, efield, n)
            probability = PsiRe**2 + PsiIm**2
            data_time.append(dt*n)
            dataRe.append(PsiRe)
            dataIm.append(PsiIm)
            dataprob.append(probability)
        
        return data_time, dataRe, dataIm, dataprob
    
    def postprocess():
        #retrieve the quantum current from the wavefunction
        pass

    # # TODO animate_field function doesn't work at all, I don't see the fields
    # def animate_field(self, t, data, source):
    #     fig, ax = plt.subplots()

    #     ax.set_xlabel("x-axis [m]")
    #     ax.set_ylabel("y-axis [m]")
    #     ax.set_xlim(0, Nx*dx)
    #     ax.set_ylim(0, Ny*dy)

    #     label = "P-field"
        
    #     ax.plot(source.x, source.y) # plot the source

    #     cax = ax.imshow(data[0])
    #     ax.set_title("T = 0")

    #     def animate_frame(i):
    #         cax.set_array(data[i])
    #         ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
    #         return cax

    #     global anim
        
    #     anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)))
    #     plt.show()

##########################################################

        


eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass
q = ct.elementary_charge 

#dy = 0.125*10**(-9)
dy = 0.125e-9
#assert(dy==dy2) # m
c = ct.speed_of_light # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = 10*dy/c


Ny = 400
Nt =10000



omega = 50e12 #[ras/s]
alpha = 4
potential = Potential(m,omega, Ny, dy)
potential.V()

Efield = ElectricField('gaussian',amplitude = 1e10)
Efield.generate(1, t0 = -5*dt, sigma = 1)


qm = QM()
res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha)
prob = res[3]
plt.plot(prob[8000])
plt.show()
#test = UCHIE()
#data_time, data = test.calc_field(dx, dy, dt, Nx, Ny, Nt, eps0, mu0, source)
#test.animate_field(data_time, data, source)