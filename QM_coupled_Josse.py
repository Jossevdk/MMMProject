import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation
import time
import psutil
#For the QM part, we require a simple 1D FDTD scheme

from uchie_FAST import Source, UCHIE

c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)

global g
g = True



# source = Source(xs, ys, 1, tc, sigma)


# scheme = UCHIE(Nx, Ny, dx, dy, dt, pml_kmax = pml_kmax, pml_nl = pml_nl)


#strategy: electric field from EM part is source in the interaction Hamiltionian. 
#Output is a quantum current which serves as a source for EM part


### Electric field ###




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
class coupled:
    def __init__(self,order, Nx,Ny, dx, dy,dt, pml_kmax, pml_nl, J0, xs, ys, tc, sigma):#,Ny,dy, dt, hbar, m, q, r, potential, efield, n,order,N):
        self.Nx = Nx
        self.xs = xs
        self.ys = ys
        self.Ny= Ny
        self.dx = dx
        self.dy=dy
        self.dt = dt
        self.pml_kmax = pml_kmax
        self.pml_nl=pml_nl
        self.J0 = J0
        self.tc = tc
        self.sigma=sigma
        self.result = None
        self.order = order
        self.potential = potential
        self.source  = Source(self.xs, self.ys, self.J0, self.tc, self.sigma)
        #print(self.source.x)
        self.uchie = UCHIE(self.Nx, self.Ny, self.dx, self.dy, self.dt, pml_kmax = self.pml_kmax, pml_nl = self.pml_nl)
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
    def update(self, PsiRe, PsiIm, dy, dt, hbar, m, q, r, potential, n,order,N, Efield):
        #E = efield.generate((n)*dt)*np.ones(Ny)
        E = Efield[2*Nx//4,:]
        Eo = E
        print(E[Ny//2])
        #print((q*r*E-potential.V())[Ny//2 - 5: Ny//2 + 5])
        #print(np.max(E))
        # print(Efield[1:, Ny//2])
        #E= 0
        PsiReo = PsiRe
        PsiRe = PsiReo -hbar*dt/(2*m)*self.diff(PsiIm,dy,order) - dt/hbar*(q*r*E-potential.V())*PsiIm
        #PsiRe = PsiRe -hbar*dt/(2*m)*self.diff(PsiIm,dy,order) - dt/hbar*(-potential.V())*PsiIm
       
        PsiRe[0] = 0
        PsiRe[-1] = 0
        #E = efield.generate((n+1/2)*dt)*np.ones(Ny)
        #E = efield.generate((n+1/2)*dt,omega=omega)*np.ones(Ny)
        #E= 0

        E = Efield[2*Nx//4,1:] + Efield[2*Nx//4,:-1]/2
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
    
        
    
    def calc_wave(self, dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N):
        PsiRe,PsiIm ,r = self.initialize(dy, Ny,m,omega, hbar,alpha)
        data_time = []
        dataRe = []
        dataIm = []
        dataprob = []
        datacurr = []

        
        data = []
        E = np.zeros(Ny)

        

        for n in range(0, Nt):
           
            Eold = E
            Efield = 5*1e13*self.uchie.Update(n,self.source)
            #E = copy.deepcopy(Efield[2*Nx//4,:])
            E = Efield[2*Nx//4,:]
            #print(Efield)
            
            #print((q*r*E-potential.V())[Ny//2 - 5: Ny//2 + 5])
            #print(np.max(E))
            # print(Efield[1:, Ny//2])
            #E= 0
            PsiReo = PsiRe
            PsiRe = PsiReo -hbar*dt/(2*m)*self.diff(PsiIm,dy,order) - dt/hbar*(q*r*E-potential.V())*PsiIm
            #PsiRe = PsiRe -hbar*dt/(2*m)*self.diff(PsiIm,dy,order) - dt/hbar*(-potential.V())*PsiIm

            PsiRe[0] = 0
            PsiRe[-1] = 0
            #E = efield.generate((n+1/2)*dt)*np.ones(Ny)
            #E = efield.generate((n+1/2)*dt,omega=omega)*np.ones(Ny)
            #E= 0

            E_half = (E + Eold)/2 
            PsiImo = PsiIm
            PsiIm = PsiImo +hbar*dt/(2*m)*self.diff(PsiRe,dy,order) + dt/hbar*(q*r*E_half-potential.V())*PsiRe
            #PsiIm = PsiIm +hbar*dt/(2*m)*self.diff(PsiRe,dy, order) + dt/hbar*(-potential.V())*PsiRe

            PsiIm[0] = 0
            PsiIm[-1] = 0
            #We need the PsiIm at half integer time steps -> interpol
            PsiImhalf = (PsiImo + PsiIm)/2
            J = N*q*hbar/(m*dy)*(PsiRe*np.roll(PsiImhalf,-1) - np.roll(PsiRe,-1)*PsiImhalf)
            J[0]=0
            J[-1]= 0

            prob = PsiRe**2  + PsiImhalf**2
            #probability = PsiRe**2 + PsiIm**2 
            # PsiReint = 1/2*(PsiRe + np.roll(PsiRe, -1))
            # PsiReint[0]=0
            # PsiReint[-1] = 0
            # probability = PsiReint**2 + PsiIm**2
            if n%10 == 0:
                print(prob[Ny//2])
                print(np.sum(prob))
                # print(PsiIm[Ny//2])
                # print((q*r*E-potential.V())[Ny//2])
                # print((q*r*E_half-potential.V())[Ny//2])
                print(n)
                data_time.append(dt*n)
                dataRe.append(PsiRe)
                dataIm.append(PsiIm)
                dataprob.append(copy.deepcopy(prob))
                datacurr.append(J)
                #data.append(copy.deepcopy((Efield.T)))
                data.append(Efield.T)
            #J in input for EM part
        self.result = data_time, dataRe, dataIm, dataprob, datacurr, data
        return data_time, dataRe, dataIm, dataprob, datacurr, data
    
    def expvalues(self,dt, dy,  type):
        if self.result == None:
            data_time, dataRe, dataIm, dataprob, datacurr, data = self.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, alpha,order,N)
        else: data_time, dataRe, dataIm, dataprob, datacurr , data = self.result
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

    def heatmap(self,dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N):
        if self.result == None:
            res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
        else:
            res = self.result
        prob = res[3]
        probsel = prob
        plt.imshow(np.array(probsel).T)
        plt.show()


    def animate(self,dy, dt, Ny, Nt,  hbar, m ,q ,potential, alpha,order,N):
        if self.result == None:
            res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
        else:
            res = self.result
        prob = res[3]
        probsel = prob
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

    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        datasel= data
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(datasel[0],vmin = -1e-13, vmax = 1e-12)
        ax.set_title("T = 0")
        # Draw a vertical line at x = Nx/2
        ax.axvline(x=2*Nx//4, color='r')  # 'r' is the color red

        def animate_frame(i):
            cax.set_array(datasel[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(datasel)), interval=20)
        plt.show()

##########################################################
dx = 1e-10 # m
dy = 0.125e-9# ms

Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0
#print(dt)

Nx = 1000
Ny = 1000
Nt = 400

pml_nl = 50
pml_kmax = 20
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)

J0 = 10e7
xs = Nx*dx/3
ys = Ny*dy/2

tc = dt*Nt/4
#print(tc)
sigma = tc/10

#######################################################
        
eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass
q = ct.elementary_charge 

#dy = 0.125*10**(-9)
# dy = 0.1
# dy = 0.1
#assert(dy==dy2) # m
c = ct.speed_of_light # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
#dt = 0.1*dy/c


# Ny = 300
# Nt =100
N = 30000 #particles/m2


omega = 50e12 #[rad/s]
alpha = 0
potential = Potential(m,omega, Ny, dy)
potential.V()

#Efield = ElectricField('gaussian',dt, amplitude = 10000000)
#Efield = ElectricField('sinusoidal',dt, amplitude = 5e6)
#Efield.generate(1)

order = 'fourth'

qm = coupled(order, Nx,Ny, dx, dy,dt, pml_kmax, pml_nl, J0, xs, ys, tc, sigma)

start_time = time.time()


res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, alpha, order,N)
# prob = res[3]
# probsel = prob[::100]


process = psutil.Process()
print("Memory usage:", process.memory_info().rss/(1024*1024), 'MB') # print memory usage
print("CPU usage:", process.cpu_percent()) # print CPU usage

end_time = time.time()


print("Execution time: ", end_time - start_time, "seconds")

qm.animate( dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
qm.animate_field(res[0], res[5])
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

# exp = qm.expvalues(dt, dy, 'continuity')
# expsel = exp[::100]
#     #print(expsel)
# plt.imshow(np.array(expsel).T)
# plt.show()


qm.heatmap(dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
