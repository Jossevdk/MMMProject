import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

M=np.array([[0,0,100,100,100,100,100,100,300,300,300,300,300,300,500,500,500,500,500,500,1000,1000,1000,1000] for i in range(0,20)]) 

def update(i):
    M[7,i] = 1000
    M[19-i,10] = 500
    matrice.set_array(M)

fig, ax = plt.subplots()
matrice = ax.matshow(M)
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=19, interval=500)
plt.show()
