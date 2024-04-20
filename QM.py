import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation

#For the QM part, we require a simple 1D FDTD scheme


#strategy: electric field from EM part is source in the interaction Hamiltionian. 
#Output is a quantum current which serves as a source for EM part


### Electric field ###
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
        t0 = 10000*self.dt
        return self.amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    def _sinusoidal(self, t, omega=1):
        t0= 1000*self.dt
        #add damping function
        return self.amplitude * np.sin(omega * t)*2/np.pi* np.arctan(t/t0)



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
    def __init__(self,order):#,Ny,dy, dt, hbar, m, q, r, potential, efield, n,order,N):
        self.result = None
        self.order = order
        self.potential = potential
       
        #self.Ny = Ny, self.dy = dy, self.dt = dt, self.hbar = hbar = self.m = m, self.q = q,
    def initialize(self, dy, Ny,m,omega, hbar,alpha):
        #coherent state at y=0 for electron
        
        self.r = np.linspace(-Ny/2*dy, Ny/2*dy,Ny)
        PsiRe = (m*omega/(np.pi*hbar))**(1/4)*np.exp(-m*omega/(2*hbar)*(self.r-alpha*np.sqrt(2*hbar/(m*omega))*np.ones(Ny))**2)
        PsiIm = np.zeros(Ny)
        return PsiRe, PsiIm, self.r

    def diff(self,psi,dy,order):
        if order == 'second':
            psi= (np.roll(psi,1) -2*psi + np.roll(psi,-1))/dy**2
            psi[0] = 0
            psi[-1] = 0
            return psi
        elif order == 'fourth':
            psi= (-np.roll(psi,2) + 16*np.roll(psi,1) -30*psi + 16*np.roll(psi,-1)-np.roll(psi,-2))/(12*dy**2)
            psi[0] = 0
            psi[1]= 0
            psi[-1] = 0
            psi[-2]=0
        else:
            raise ValueError(f"Order schould be 'second' or 'fourth'")
        
        return psi

    ### Update ###
    def update(self, PsiRe, PsiIm, dy, dt, hbar, m, q, r, potential, efield, n,order,N):
        #E = efield.generate((n)*dt)*np.ones(Ny)
      
        E = efield.generate((n)*dt, omega=omega)*np.ones(Ny)

        print(E[Ny//2])

        #E= 0
        PsiReo = PsiRe
        PsiRe = PsiReo -hbar*dt/(2*m)*self.diff(PsiIm,dy,order) - dt/hbar*(q*r*E-potential.V())*PsiIm
        #PsiRe = PsiRe -hbar*dt/(2*m)*self.diff(PsiIm,dy,order) - dt/hbar*(-potential.V())*PsiIm
       
        PsiRe[0] = 0
        PsiRe[-1] = 0
        #E = efield.generate((n+1/2)*dt)*np.ones(Ny)
        E = efield.generate((n+1/2)*dt,omega=omega)*np.ones(Ny)
        #E= 0
        PsiImo = PsiIm
        PsiIm = PsiImo +hbar*dt/(2*m)*self.diff(PsiRe,dy,order) + dt/hbar*(q*r*E-potential.V())*PsiRe
        #PsiIm = PsiIm +hbar*dt/(2*m)*self.diff(PsiRe,dy, order) + dt/hbar*(-potential.V())*PsiRe
        
        PsiIm[0] = 0
        PsiIm[-1] = 0
        #We need the PsiIm at half integer time steps -> interpol
        PsiImhalf = (PsiImo + PsiIm)/2
        J = N*q*hbar/(m*dy)*(PsiRe*np.roll(PsiImhalf,-1) - np.roll(PsiRe,-1)*PsiImhalf)
        J[0]=0
        J[-1]= 0

        prob = PsiRe**2  + PsiImhalf**2
        
        return PsiRe, PsiIm, J, prob
    
        
    
    def calc_wave(self, dy, dt, Ny, Nt,  hbar, m ,q ,potential, efield,alpha,order,N):
        PsiRe,PsiIm ,r = self.initialize(dy, Ny,m,omega, hbar,alpha)
        data_time = []
        dataRe = []
        dataIm = []
        dataprob = []
        datacurr = []

        for n in range(1, Nt):
            PsiRe, PsiIm , J ,prob = self.update(PsiRe, PsiIm, dy, dt, hbar, m, q, r, potential, efield, n,order,N)
            #probability = PsiRe**2 + PsiIm**2 
            # PsiReint = 1/2*(PsiRe + np.roll(PsiRe, -1))
            # PsiReint[0]=0
            # PsiReint[-1] = 0
            # probability = PsiReint**2 + PsiIm**2
            data_time.append(dt*n)
            dataRe.append(PsiRe)
            dataIm.append(PsiIm)
            dataprob.append(prob)
            datacurr.append(J)
            #J in input for EM part
        self.result = data_time, dataRe, dataIm, dataprob, datacurr
        return data_time, dataRe, dataIm, dataprob, datacurr
    
    def expvalues(self,dt, dy,  type):
        if self.result == None:
            data_time, dataRe, dataIm, dataprob, datacurr = self.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)
        else: data_time, dataRe, dataIm, dataprob, datacurr = self.result
        if type == 'position':
            exp = []
            for el in dataprob:
                exp.append(np.sum(el*self.r*dy))
            
        if type == 'momentum':
            exp = []
            for i in range(len(data_time)-1):
                val = 1/2*((1/2*(np.roll(dataRe[i+1], -1) + np.roll(dataRe[i],-1)) - 1j*np.roll(dataIm[i],-1)) + (1/2*(dataRe[i+1]+ dataRe[i]) - 1j*dataIm[i]))*((-1j*hbar/(2*dy)*(np.roll(dataRe[i+1],-1) +np.roll(dataRe[i],-1)-dataRe[i+1]- dataRe[i]))+hbar/dy*(np.roll(dataIm[i],-1)-dataIm[i]))
                exp.append(np.sum(val))

        if type == 'energy':
            exp= []
            for i in range(1,len(data_time)):
                #fix psire because not at right moment
                dataIm[i] = 1/2*(dataIm[i-1] + dataIm[i])
                dataIm[i][0]= 0
                dataIm[i][-1] = 0
                val = (dataRe[i]-1j*dataIm[i])*(-hbar**2/(2*m)*self.diff(dataRe[i]+1j*dataIm[i], dy, order)+self.potential.V()*(dataRe[i]+1j*dataIm[i]))
                exp.append(np.sum(val))

        if type == 'continuity':
            exp = []
            for i in range(1,len(data_time)-1):
                #rho is know at n+1/2, r, J is known at n+1/2, r+1/2

                #find curr at n trough interpol:
                datacurr[i] = 1/2* (datacurr[i] + datacurr[i-1])
             
                #for rho deriv in time puts time also at n, for J deriv puts pos at r
                val = q*(dataprob[i]-dataprob[i-1])/dt+1/dy*(datacurr[i]- np.roll(datacurr[i],1))
                exp.append(val[1:-1])
                #exp.append(np.sum(val[1:-1]))

        return exp

    
    def postprocess():
        
        pass

    def heatmap (self,dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N):
        if self.result == None:
            res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)
        else:
            res = self.result
        prob = res[3]
        probsel = prob[::100]
        plt.imshow(np.array(probsel).T)
        plt.show()


    def animate(self,dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N):
        if self.result == None:
            res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)
        else:
            res = self.result
        prob = res[3]
        probsel = prob[::100]
        fig, ax = plt.subplots()

        # Create an empty plot object
        line, = ax.plot([], [])

        def initanim():
            ax.set_xlim(0, len(probsel[0]))  # Adjust the x-axis limits if needed
            ax.set_ylim(0, np.max(probsel))  # Adjust the y-axis limits if needed
            return line,

        # Define the update function
        def updateanim(frame):
            line.set_data(np.arange(len(probsel[frame])), probsel[frame])
            return line,


      
        anim = FuncAnimation(fig, updateanim, frames=len(probsel), init_func=initanim, interval = 30)

        # Show the animation
        plt.show()

##########################################################

        
eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass
q = ct.elementary_charge 

#dy = 0.125*10**(-9)
dy = 0.125e-9
#dy = 0.1
#assert(dy==dy2) # m
c = ct.speed_of_light # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = 10*dy/c


Ny = 400
#Nt =20000
Nt = 100000
N = 10000 #particles/m2


omega = 50e12 #[rad/s]
alpha = 0
potential = Potential(m,omega, Ny, dy)
potential.V()

#Efield = ElectricField('gaussian',dt, amplitude = 10000000)
Efield = ElectricField('sinusoidal',dt, amplitude = 1e7)
#Efield.generate(1)

order = 'fourth'

qm = QM(order)
res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha, order,N)
# prob = res[3]
# probsel = prob[::100]

qm.animate( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)
# plt.imshow(probsel)
# plt.colorbar()
# #plt.plot(prob[8000])
# plt.show()
# types = ['position', 'momentum', 'energy']
# for type in types: 
#     exp = qm.expvalues(dt, dy, type)
#     expsel = exp[::100]
#     #print(expsel)
#     plt.plot(expsel)
#     plt.title(type)
#     plt.show()

exp = qm.expvalues(dt, dy, 'continuity')
expsel = exp[::100]
    #print(expsel)
plt.imshow(np.array(expsel).T)
plt.show()


qm.heatmap(dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)
