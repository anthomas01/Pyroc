#Imports
from pyroc import CSTAirfoil2D
import numpy as np
import matplotlib.pyplot as plt

##Plot Airfoil example
#Initialize array for airfoil coordinates
x = np.linspace(0,1.0,1000)
z = np.zeros_like(x)
arr = np.array(list(zip(x,z)))

#Upper surface
cst = CSTAirfoil2D(arr,order=7) #Initialize upper surface
cst.updatePsiZeta() #Create psi from x coords and calc zeta from coeffs
xzU = cst.updateCoords() #Update coordinates from parameterized values
plt.plot(xzU[:,0],xzU[:,1],'-',label='Upper') #Plot coordinates

#Lower Surface
cstLower = CSTAirfoil2D(arr,order=7,shapeScale=-1.0) #Shape coeffs initialized to -1 for lower surface
cstLower.updatePsiZeta()
xzL = cstLower.updateCoords()
plt.plot(xzL[:,0],xzL[:,1],'-',label='Lower')

fig = plt.subplot()
plt.title('Base Airfoil Shape - '+str(cst.getCoeffs()))
plt.legend()
plt.show()