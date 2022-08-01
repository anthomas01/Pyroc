from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import stl
from stl import mesh

##Plot Wing ex: ## Origin at LE of root chord
# rootChord = 1.0
# tipChord = 1.0 * rootChord
# span = 6.0 * rootChord

# xVals = np.linspace(0,rootChord,80)
# yVals = np.linspace(0,span,50)
# xv,yv = np.meshgrid(xVals,yVals)
# xv = xv.flatten()
# yv = yv.flatten()
# psiVals = np.linspace(0,1,len(xVals))
# etaVals = np.linspace(0,1,len(yVals))
# psiv, etav = np.meshgrid(psiVals,etaVals)
# psiv = psiv.flatten()
# etav = etav.flatten()
# surface1 = np.array(list(zip(xv,yv,np.zeros(len(xv)))))
# surface2 = np.array(list(zip(xv,yv,np.zeros(len(xv)))))
# tri = Delaunay(np.array([xv,yv]).T)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# #Airfoil
# # airfoilSS = CSTAirfoil3D(surface1)
# # airfoilSS.updatePsiEtaZeta()
# # airfoilSS.updateCoords()
# # airfoilPS = CSTAirfoil3D(surface2, zScale=-1.0)
# # airfoilPS.updatePsiEtaZeta()
# # airfoilPS.updateCoords()
# # coords = {'upper': [airfoilSS.surface],
# #           'lower': [airfoilPS.surface]}

# #Wing
# sweepAngle = np.radians(5.0)
# twistAngle = np.radians(5.0)
# refAxes = np.array([[1.0,0.0,0.0],[np.sin(sweepAngle),np.cos(sweepAngle),0.0]])

# def wingExtFunc(eta, *coeffs):
#     coeffs = coeffs[0]
#     return coeffs[0]*np.power(eta,coeffs[1])

# def wingExtModFunc(pts, eta, *coeffs):
#     coeffs = coeffs[0]
#     def angle(eta, *coeffs):
#         coeffs = coeffs[0]
#         return (np.exp(10*eta)-1)*coeffs[0] 
#     newPts = []
#     for _ in range(len(pts)):
#         pt = pts[_]
#         pt[2] += -coeffs[-1]*np.log(np.cos(angle(eta,coeffs)))
#         newPts.append(pt)
#     return np.array(newPts)

# def wingChordModFunc(eta, *coeffs):
#     coeffs = coeffs[0]
#     return (rootChord-eta*(rootChord-coeffs[0]))

# def wingTwistFunc(pts, eta, *coeffs):
#     coeffs = coeffs[0]
#     def angle(eta, *coeffs):
#         coeffs = coeffs[0]
#         return eta*coeffs[0]
#     newPts = np.zeros([len(pts),3])
#     for _ in range(len(pts)):
#         pt = pts[_]
#         rotV = rotation.rotVbyW(pt,np.array([0.0,1.0,0.0]),angle(eta,coeffs))
#         newPts[_,:] = rotV
#     return newPts

# wingSS = CSTWing3D(surface1, extrudeFunc=wingExtFunc, extClassCoeffs=[2.0,3.0], extModFunc=wingExtModFunc, extModCoeffs=[3e-5, 5.0], chordModFunc=wingChordModFunc,
#                    chordModCoeffs=[1e-1*rootChord], csModFunc=wingTwistFunc, csModCoeffs=[twistAngle], refAxes=refAxes)
# wingSS.setPsiEtaZeta(psiVals=psiv,etaVals=etav)
# wingSS.updateZeta()
# wingSS.updateCoords()
# wingPS = CSTWing3D(surface2, extrudeFunc=wingExtFunc, extClassCoeffs=[2.0,3.0], extModFunc=wingExtModFunc, extModCoeffs=[3e-5, 5.0], chordModFunc=wingChordModFunc,
#                    chordModCoeffs=[1e-1*rootChord], csModFunc=wingTwistFunc, csModCoeffs=[twistAngle], refAxes=refAxes, zScale=-1.0)
# wingPS.setPsiEtaZeta(psiVals=psiv,etaVals=etav)
# wingPS.updateZeta()
# wingPS.updateCoords()
# coords = {'upper': [wingSS.surface],
#           'lower': [wingPS.surface]}

# for key in coords.keys():
#     for coordSet in coords[key]:
#         #ax.scatter(coordSet[:,0],coordSet[:,1],coordSet[:,2])
#         ax.plot_trisurf(coordSet[:,0], coordSet[:,1], coordSet[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.axes.set_xlim3d(left=-span/2, right=span/2) 
# ax.axes.set_ylim3d(bottom=0, top=span) 
# ax.axes.set_zlim3d(bottom=-span/2, top=span/2)
# plt.show()