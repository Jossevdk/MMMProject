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
    def __init__(self, field_type, dt, Nt, omega = 1, amplitude=1.0):
        self.field_type = field_type
        self.amplitude = amplitude
        self.dt = dt
        self.omega = omega
        self.Nt = Nt
        self.t0 = self.Nt*self.dt/5
        self.sigma = self.t0/5

    def generate(self, t, **kwargs):
        if self.field_type == 'gaussian':
            return self._gaussian(t)
        elif self.field_type == 'sinusoidal':
            return self._sinusoidal(t, **kwargs)
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")

    def _gaussian(self, t):
        return self.amplitude * np.exp(-0.5 * ((t - self.t0) / self.sigma) ** 2)

    def _sinusoidal(self, t):
        #add damping function
        return self.amplitude * np.cos(self.omega * t)*2/np.pi* np.arctan(t/self.t0)
    
class vecpot:
    def __init__(self, dt, Nt, omega , amplitude=1.0):
     
        self.amplitude = amplitude
        self.dt = dt
        self.omega = omega
        self.Nt = Nt
        self.t0 = self.Nt*self.dt/5

    def generate(self, t):
        return -self.amplitude/self.omega * np.sin(self.omega * t)*2/np.pi* np.arctan(t/self.t0)



### Potential ###
class Potential:
    def __init__(self, m, omega,Ny,dy):
        self.m = m
        self.omega = omega
        self.Ny = Ny
        self.dy = dy
        
        
    #This will call the function depending on which type of source you have    
    def V(self):
        V = 0.5*self.m*self.omega**2* (np.linspace(-self.Ny//2*self.dy, self.Ny//2*self.dy,self.Ny))**2
        return V
    

#### QM ####
class QM:
    def __init__(self,order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omega, N, gauge, omegafield = 1, amplitude = 1, field_type = None):
        self.Ny = Ny
        self.Nt = Nt
        self.dy = dy
        self.dt = dt
        self.hbar = hbar
        self.m = m
        self.q = q
        self.alpha=alpha
        self.result = None
        self.order = order
        self.potential = potential
        #self.efield = efield
        self.omega =omega
        self.N=N
        self.gauge = gauge
        if gauge == 'length':
            if field_type ==  'gaussian':

                self.efield= ElectricField('gaussian',dt, Nt, amplitude = amplitude)

            elif field_type == 'sinusoidal':
                self.efield = ElectricField('sinusoidal',dt, Nt, omega = omegafield, amplitude = amplitude)
            else:
                raise ValueError(f"If length gauge is chosen, please provide either gaussian or sinusoidal as field type")
        if gauge == 'velocity':
            self.pot = vecpot(self.dt,self.Nt, omegafield , amplitude=amplitude)
        
        
       
    #    
    #    #coherent state at y=0 for electron
        
        self.r = np.linspace(-self.Ny/2*self.dy, self.Ny/2*self.dy,self.Ny)
        #print(self.r.shape)
        self.PsiRe = (self.m*self.omega/(np.pi*self.hbar))**(1/4)*np.exp(-self.m*self.omega/(2*self.hbar)*(self.r-self.alpha*np.sqrt(2*self.hbar/(self.m*self.omega))*np.ones(self.Ny))**2)
        self.PsiIm = np.zeros(self.Ny)
        self.J = np.zeros(self.Ny)
        
        #print(self.PsiRe)
        #self.J = 0
        #return self.PsiRe, self.PsiIm, self.r
        self.data_prob= []
        self.data_time = []
        self.data_mom=[]
        self.data_energy= []
        self.beamenergy = []

    def diff(self,psi):
        if self.order == 'second':
            psi= (np.roll(psi,1) -2*psi + np.roll(psi,-1))/self.dy**2
            psi[0] = 0
            psi[-1] = 0
            return psi
        elif self.order == 'fourth':
            psi= (-np.roll(psi,2) + 16*np.roll(psi,1) -30*psi + 16*np.roll(psi,-1)-np.roll(psi,-2))/(12*self.dy**2)
            psi[0] = 0
            psi[1]= 0
            psi[-1] = 0
            psi[-2]=0
        else:
            raise ValueError(f"Order schould be 'second' or 'fourth'")
        
        return psi

    ### Update ###
    def update(self,n):
        
        E = self.efield.generate((n)*self.dt)*np.ones(self.Ny)

        PsiReo = self.PsiRe
        self.PsiRe = PsiReo -self.hbar*self.dt/(2*self.m)*self.diff(self.PsiIm) - self.dt/self.hbar*(self.q*self.r*E-self.potential.V())*self.PsiIm
        
        
        self.PsiRe[0] = 0
        self.PsiRe[-1] = 0
        
        E = self.efield.generate((n+1/2)*self.dt)*np.ones(self.Ny)
        
        PsiImo = self.PsiIm
        self.PsiIm = PsiImo +self.hbar*self.dt/(2*self.m)*self.diff(self.PsiRe)+ self.dt/self.hbar*(self.q*self.r*E-self.potential.V())*self.PsiRe
        
        
        self.PsiIm[0] = 0
        self.PsiIm[-1] = 0
        
        #We need the PsiIm at half integer time steps -> interpol
        PsiImhalf = (PsiImo + self.PsiIm)/2
        self.J = self.hbar/(self.m*self.dy)*(self.PsiRe*np.roll(PsiImhalf,-1) - np.roll(self.PsiRe,-1)*PsiImhalf)
        self.J[0]=0
        self.J[-1]= 0
        self.J *= self.q*self.N
        

        Psi = self.PsiRe+ 1j*PsiImhalf
        momentum = np.conj(Psi)*-1j*self.hbar*1/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1))
        momentum[0] = 0
        momentum[-1] = 0

        prob = self.PsiRe**2  + PsiImhalf**2

        #energy = np.sum((self.PsiRe-1j*self.PsiIm)*(-self.hbar**2/(2*self.m)*self.diff(self.PsiRe+1j*self.PsiIm)+self.potential.V())*(self.PsiRe+1j*self.PsiIm))
        #energy = np.sum((self.PsiRe-1j*self.PsiIm)*((-self.hbar**2/(2*self.m)*self.diff(self.PsiRe+1j*self.PsiIm))+self.potential.V())*(self.PsiRe+1j*self.PsiIm))
        energy = np.sum((self.PsiRe-1j*self.PsiIm)*(-self.hbar**2/(2*self.m)*self.diff(self.PsiRe+1j*self.PsiIm)+self.potential.V()*(self.PsiRe+1j*self.PsiIm)))
        #energy = np.sum((self.PsiRe-1j*self.PsiIm)*(-self.hbar**2/(2*self.m)*self.diff(self.PsiRe+1j*self.PsiIm)*(self.PsiRe+1j*self.PsiIm))) #+ (self.PsiRe-1j*self.PsiIm)*(self.potential.V())*(self.PsiRe+1j*self.PsiIm))
        #energy = np.sum((np.conj(Psi))*(-self.hbar**2/(2*self.m)*self.diff(Psi)*(Psi)) +np.conj(Psi)*self.potential.V()*Psi)
        beam_energy = np.sum(np.conj(Psi)*(-self.q*self.r *E*(Psi)))

        self.data_time.append(n*self.dt)
        self.data_prob.append(prob)
        self.data_mom.append(np.sum(momentum))
        self.data_energy.append(energy)
        self.beamenergy.append(beam_energy)

    def update_vel(self, n):
       
        a = self.pot.generate(n*dt)

        
        PsiReo = self.PsiRe
        self.PsiRe = PsiReo -self.hbar*self.dt/(2*self.m)*self.diff(self.PsiIm) + self.dt/self.hbar*(self.potential.V())*self.PsiIm +self.dt*self.q/self.m*a/(2*self.dy)*(np.roll(self.PsiRe,-1)-np.roll(self.PsiRe,1))
        
        
        self.PsiRe[0] = 0
        self.PsiRe[-1] = 0

        a = self.pot.generate((n+1/2)*dt)
        
        PsiImo = self.PsiIm
        self.PsiIm = PsiImo +self.hbar*self.dt/(2*self.m)*self.diff(self.PsiRe) - self.dt/self.hbar*(self.potential.V())*self.PsiRe +self.dt*self.q/self.m*a/(self.dy*2)*(np.roll(self.PsiIm,-1)-np.roll(self.PsiIm,1))
        
        
        self.PsiIm[0] = 0
        self.PsiIm[-1] = 0
        #We need the PsiIm at half integer time steps -> interpol
        PsiImhalf = (PsiImo + self.PsiIm)/2
        self.J = self.hbar/(self.m*self.dy)*(self.PsiRe*np.roll(PsiImhalf,-1) - np.roll(self.PsiRe,-1)*PsiImhalf)
        self.J[0]=0
        self.J[-1]= 0


        Psi = self.PsiRe+ 1j*PsiImhalf
        momentum = np.conj(Psi)*(-1j*self.hbar*1/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1)) - self.q*a*Psi)
        momentum[0] = 0
        momentum[-1] = 0

        prob = self.PsiRe**2  + PsiImhalf**2
        energy = np.sum((self.PsiRe-1j*self.PsiIm)*(-self.hbar**2/(2*self.m)*self.diff(self.PsiRe+1j*self.PsiIm)+self.potential.V()*(self.PsiRe+1j*self.PsiIm))+ np.conj(Psi)*1j*self.hbar*q/(self.m)*a/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1)))

        self.data_time.append(n*self.dt)
        self.data_prob.append(prob)
        self.data_mom.append(np.sum(momentum))
        self.data_energy.append(energy)
    


    def calcwave(self):
        if self.gauge == 'length':
            for n in range (self.Nt):
                self.update(n)

        elif self.gauge == 'velocity':
            for n in range (self.Nt):
                self.update_vel(n)

    
    def expvalues(self,type):
        
        if type == 'position':
            exp = []
            for el in self.data_prob:
                exp.append(np.sum(el*self.r*self.dy))
            return(exp)
            # plt.plot(exp)
            # plt.show()            
        if type == 'momentum':
            plt.plot(self.data_mom)
            plt.show()
            

        if type == 'energy':
            plt.plot(self.beamenergy)
            plt.plot(self.data_energy)
            plt.show()
    

        
        # if type == 'continuity':
        #     exp = []
        #     for i in range(1,len(data_time)-1):
        #         #rho is know at n+1/2, r, J is known at n+1/2, r+1/2

        #         #find curr at n trough interpol:
        #         datacurrhalf = 1/2* (datacurr[i] + datacurr[i-1])
                
        #         #for rho deriv in time puts time also at n, for J deriv puts pos at r
        #         val = (dataprob[i] - dataprob[i-1])[1:]/dt+(datacurrhalf- np.roll(datacurrhalf,1))[1:]/dy
        #         exp.append(val)
        #         #exp.append(np.sum(val[1:-1]))
        # if type == 'J':
        #     exp = []
        #     for i in range(1,len(data_time)-1):
        #         #rho is know at n+1/2, r, J is known at n+1/2, r+1/2

        #         #find curr at n trough interpol:
        #         datacurrhalf = 1/2* (datacurr[i] + datacurr[i-1])
                
        #         #for rho deriv in time puts time also at n, for J deriv puts pos at r
        #         val = -(datacurrhalf- np.roll(datacurrhalf,1))[1:]/dy
        #         exp.append(val)
        #         #exp.append(np.sum(val[1:-1]))
        # if type == 'dens':
        #     exp = []
        #     for i in range(1,len(data_time)-1):
        #         #rho is know at n+1/2, r, J is known at n+1/2, r+1/2

        #         #find curr at n trough interpol:
                
        #         #for rho deriv in time puts time also at n, for J deriv puts pos at r
        #         val = (dataprob[i] - dataprob[i-1])[1:]/dt
        #         exp.append(val)
        #         #exp.append(np.sum(val[1:-1]))

        #return exp



    
  

    # def heatmap (self,dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N):
    #     if self.result == None:
    #         res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)
    #     else:
    #         res = self.result
    #     prob = res[3]
    #     probsel = prob[::100]
    #     plt.imshow(np.array(probsel).T)
    #     plt.show()


    def animate(self):
       
        self.data_probsel = self.data_prob[::50]
        self.data_timesel= self.data_time[::50]
        fig, ax = plt.subplots()

       
        line, = ax.plot([], [])

        def initanim():
            ax.set_xlim(0, len(self.data_probsel[0]))  
            ax.set_ylim(0, np.max(self.data_probsel))  
            return line,

        # Define the update function
        def updateanim(frame):
            line.set_data(np.arange(len(self.data_probsel[frame])), self.data_probsel[frame])
            return line,


      
        anim = FuncAnimation(fig, updateanim, frames=len(self.data_timesel), init_func=initanim, interval = 20)

    
        plt.show()

##########################################################


# def animate_curve(data):
#     fig, ax = plt.subplots()
#     xdata, ydata = [], []
#     ln, = plt.plot([], [], 'r-')
#     def init():
#         ax.set_xlim(0, len(data[0]))
#         ax.set_ylim(np.min(data), np.max(data))
#         return ln,

#     def update(frame):
#         xdata=np.arange(len(data[frame]))
#         ydata=data[frame]
#         ln.set_data(xdata, ydata)
#         return ln,

#     ani = FuncAnimation(fig, update, frames=len(data), init_func=init, interval=30)
#     plt.show()

# def animate_two_curves(data1, data2):
#     fig, ax = plt.subplots()
#     xdata, ydata = [], []
#     ln, = plt.plot([], [], 'r-')
#     ln2, = plt.plot([], [], 'b-')
#     def init():
#         ax.set_xlim(0, len(data1[0]))
#         ax.set_ylim(np.min(data1), np.max(data1))
#         return ln, ln2

#     def update(frame):
#         xdata=np.arange(len(data1[frame]))
#         ydata=data1[frame]
#         ln.set_data(xdata, ydata)
#         ydata=data2[frame]
#         ln2.set_data(xdata, ydata)
#         return ln, ln2

#     ani = FuncAnimation(fig, update, frames=len(data1), init_func=init, interval=30)
#     plt.show()

eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass*0.15
q = -ct.elementary_charge 

#dy = 0.125*10**(-9)
dy = 0.25e-10
#dy = 0.1
#assert(dy==dy2) # m
c = ct.speed_of_light # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c


Ny = 500
#Nt =20000
Nt = 30000
N = 1 #particles/m2


omegaHO = 50e14#*2*np.pi #[rad/s]
alpha = 0

potential = Potential(m,omegaHO, Ny, dy)
potential.V()

gauge = 'velocity'
amplitude = 1e8
field_type = 'sinusoidal'
order = 'fourth'

QMscheme1 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme1.calcwave()
#QMscheme1.animate()

gauge = 'length'
amplitude = 1e8
field_type = 'sinusoidal'
order = 'fourth'

QMscheme2 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme2.calcwave()





fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
fig.suptitle('Comparison between the Length an Velocity gauge')
ax1.plot(QMscheme2.data_time[::50], QMscheme2.expvalues('position')[::50])
ax1.plot(QMscheme1.data_time[::50], QMscheme1.expvalues('position')[::50])
ax1.set_title('Position')

ax2.plot(QMscheme2.data_time[::50], QMscheme2.data_mom[::50])
ax2.plot(QMscheme1.data_time[::50], QMscheme1.data_mom[::50])
ax2.set_title('Momentum')

ax3.plot(QMscheme2.data_time[::50], QMscheme2.data_energy[::50])
ax3.plot(QMscheme1.data_time[::50], QMscheme1.data_energy[::50])
ax3.set_title("Kinetic + Potential Energy")
plt.show()


# #########################################

gauge = 'length'
amplitude = 1e8
field_type = 'sinusoidal'
order = 'second'

QMscheme3 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme3.calcwave()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
fig.suptitle('Comparison between second and fourth order')
ax1.plot(QMscheme3.data_time[::50], QMscheme3.expvalues('position')[::50])
ax1.plot(QMscheme2.data_time[::50], QMscheme2.expvalues('position')[::50])
ax1.set_title('Position')

ax2.plot(QMscheme3.data_time[::50], QMscheme3.data_mom[::50])
ax2.plot(QMscheme2.data_time[::50], QMscheme2.data_mom[::50])
ax2.set_title('Momentum')

ax3.plot(QMscheme3.data_time[::50], QMscheme3.data_energy[::50])
ax3.plot(QMscheme2.data_time[::50], QMscheme2.data_energy[::50])
ax3.set_title("Kinetic + Potential Energy")
plt.show()


gauge = 'length'
amplitude = 1e9
field_type = 'gaussian'
order = 'fourth'

QMscheme4 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme4.calcwave()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 3))
fig.suptitle('Results for a gaussian pulse in the length gauge')
ax1.plot(QMscheme4.data_time[::50], QMscheme4.expvalues('position')[::50])

ax1.set_title('Position')

ax2.plot(QMscheme4.data_time[::50], QMscheme4.data_mom[::50])

ax2.set_title('Momentum')

ax3.plot(QMscheme4.data_time[::50], QMscheme4.data_energy[::50])

ax3.set_title("Kinetic + Potential Energy")
plt.tight_layout()
plt.show()


plt.plot(QMscheme4.beamenergy)
plt.show()
# # plt.imshow(probsel)
# # plt.colorbar()
# # #plt.plot(prob[8000])
# # plt.show()
# types = ['position', 'momentum', 'energy']
# for type in types: 
#     exp = qm.expvalues(dt, dy, type)
#     expsel = exp[::100]
#     #print(expsel)
#     plt.plot(expsel)
#     plt.title(type)
#     plt.show()

# div_current = qm.expvalues(dt, dy, 'J')[::100]
# diff_density = qm.expvalues(dt, dy, 'dens')[::100]

# animate_two_curves(div_current, diff_density)
#     #print(expsel)

# expsel = qm.expvalues(dt, dy, 'continuity')[::100]
# #animate_curve(expsel)
# #plt.imshow(np.array(expsel).T)

# plt.show()


# qm.heatmap(dy, dt, Ny, Nt,  hbar, m ,q ,potential, Efield,alpha,order,N)