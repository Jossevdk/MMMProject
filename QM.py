import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 


#For the QM part, we require a simple 1D FDTD scheme


#strategy: electric field from EM part is source in the interaction Hamiltionian. 
#Output is a quantum current which serves as a source for EM part

