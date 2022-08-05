#Imports
from pyroc import *
import numpy as np
import matplotlib.pyplot as plt

##########################################################################

##Plot Airfoil example
#Initialize array for airfoil coordinates
# x = np.linspace(0,1.0,100)
# z = np.zeros_like(x)
# arr = np.array(list(zip(x,z)))

# #Upper surface
# cst = cst2d.CSTAirfoil2D(arr) #Initialize upper surface
# xzU = cst.updateCoords() #Update coordinates from parameterized values
# plt.plot(xzU[:,0],xzU[:,1],'-',label='Upper') #Plot coordinates

# #Lower Surface
# cstLower = cst2d.CSTAirfoil2D(arr,shapeScale=-1.0) #Shape coeffs initialized to -1 for lower surface
# xzL = cstLower.updateCoords()
# plt.plot(xzL[:,0],xzL[:,1],'-',label='Lower')

# plt.title('Base Airfoil Shape - '+str(cst.getCoeffs()))
# plt.legend()
# plt.show()

##########################################################################

##Test curve fit example
#Initialize array
# x = np.linspace(0,1.0,100)
# z = -np.power(x-0.5,2)+0.25+np.sin(x)-np.sqrt(x) #Random curve to try to fit
# arr = np.array(list(zip(x,z)))

# #Fit Surface
# cst = cst2d.CSTAirfoil2D(arr,order=7) #Initialize surface
# cst.fit2d()
# xz = cst.origCoords
# xzNew = cst.updateCoords()
# plt.plot(xz[:,0],xz[:,1],'-',label='Orig') #Plot coordinates
# plt.plot(xzNew[:,0],xzNew[:,1],'-',label='Fit') #Plot coordinates

# plt.title('CST fit to curve - '+str([round(_,3) for _ in cst.getCoeffs()]))
# plt.legend()
# plt.show()

##########################################################################

#Design GUI Example
x = np.linspace(0,1.0,100)
z = np.zeros_like(x)
arr = np.array(list(zip(x,z)))
upper = CSTAirfoil2D(arr,order=3)
lower = CSTAirfoil2D(arr,order=3,shapeScale=-1.0)
geo = GeoEx(surfaces=[upper, lower])

p = PyrocDesign(geo)
p.root.mainloop()
