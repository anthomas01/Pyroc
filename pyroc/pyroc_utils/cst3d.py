#from norm import *
from collections import OrderedDict
from cst2d import * #Currently plotting
import numpy as np
import scipy.optimize as scp
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import stl
from stl import mesh
from curve import *

class CST3DParam():
    """
    Class for storing cst parameterization information in 3d

    Closed surfaces must be split over valid cartesian domains
    TODO Implement alternative coordinate spaces

    Parameters
    ----------
    surface : ndarray
        Boundary surface to parameterize
        must be x,y,z=[N,3]

    extrudeFunc: Function
        Function for offsetting position of cross section origins
        must be [xOff,yOff,zOff]=f(eta)

    extModFunc: Function
        Function for modifying points of extrudeFunc curve

    classFunc: Function
        Function for determining cross section shape
        must be [zeta]=f([psi])

    csModFunc: Function
        Function for modifying cross section shapes along extrusion
        must be newPts = f(pts,eta,*coeffs)

    TODO multiple shape/class funcs along extrusion?

    refAxis: list [2,3]
        Axis to do 2d param in, followed by extrusion reference axis
        must be [[x,y,z],[x,y,z]]

    Order: list [int,int]
        Order of class function followed by order of extrusion equation
    
    TODO Error checking due to:
        everything
    """

    def __init__(self, surface, extrudeFunc=None, extClassCoeffs=[], extShapeCoeffs=[], extModFunc=None, extModCoeffs=[],
                 classFunc=None, csClassCoeffs=[], csShapeCoeffs=[], csModFunc=None, csModCoeffs=[], chordModFunc=None,
                 chordModCoeffs=[], origin=np.array([0.0,0.0,0.0]), refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), order=[5,0], zScale=1.0):

        self.origSurface = surface
        self.transformedPoints = surface
        self.surface = self.origSurface
        self.psiAx = refAxes[0,:]/np.linalg.norm(refAxes[0,:])
        self.etaAx = refAxes[1,:]/np.linalg.norm(refAxes[1,:])
        self.zetaAx = np.cross(self.psiAx,self.etaAx)
        self.updateTransformedPoints(self.origSurface)
        self.refLen = np.max(self.transformedPoints[:,1])-np.min(self.transformedPoints[:,1])
        self.zScale = zScale

        self.numPts = len(surface[:,0])
        self.origin = origin
        self.psiEtaZeta = np.zeros([self.numPts,3])

        #sortedOnPsi = np.array(sorted(self.origSurface, key=lambda x:np.dot(x,self.psiAx)))
        #self.leCurve = np.array([])
        #for i in range(len(sortedOnPsi))[1:]:
        #    psiErr = np.linalg.norm(sortedOnPsi[i]-sortedOnPsi[i-1])
        #    if psiErr<tol:

        self.extrudeFunc = self.defaultExtrudeFunc if extrudeFunc==None else extrudeFunc
        self.extModFunc = self.defaultExtModFunc if extModFunc==None else extModFunc
        self.classFunc = self.defaultClassFunc if classFunc==None else classFunc
        self.csModFunc = self.defaultCsModFunc if csModFunc==None else csModFunc
        self.chordModFunc = self.defaultChordModFunc if chordModFunc==None else chordModFunc
        
        self.order = order

        if (len(extShapeCoeffs)==self.order[1]+2) or (len(extShapeCoeffs)==self.order[1]+1):
            self.extShapeCoeffs = extShapeCoeffs
        else:    
            self.extShapeCoeffs = [1 for _ in range(order[1]+1)]

        if (len(csShapeCoeffs)==self.order[0]+2) or (len(csShapeCoeffs)==self.order[0]+1):
            self.csShapeCoeffs = csShapeCoeffs
        else:    
            self.csShapeCoeffs = [1 for _ in range(order[0]+1)]

        self.extClassCoeffs = extClassCoeffs
        self.csClassCoeffs = csClassCoeffs    
        self.extModCoeffs = extModCoeffs
        self.csModCoeffs = csModCoeffs
        self.chordModCoeffs = chordModCoeffs

        self.extOffset = 0.0
        self.csOffset = 0.0

        self.extParam = CST2DParam(coords=np.array([[0,0]]), classFunc=self.extrudeFunc, order=self.order[1],
            classCoeffs=self.extClassCoeffs, shapeCoeffs=self.extShapeCoeffs, shapeOffset=self.extOffset)
        self.extParam.refLen = self.refLen
        self.csParam = CST2DParam(coords=np.array([[0,0]]), classFunc=self.classFunc, order=self.order[0], 
            classCoeffs=self.csClassCoeffs, shapeCoeffs=self.csShapeCoeffs, shapeOffset=self.csOffset, zScale=self.zScale)

    def updatePsiEtaZeta(self):
        self.psiEtaZeta = self.calcPsiEtaZeta(self.surface)

    def setPsiEtaZeta(self, psiVals=None, etaVals=None, zetaVals=None):
        v = psiVals if psiVals is not None else etaVals if etaVals is not None else zetaVals
        if len(self.psiEtaZeta)!=len(v):
            self.psiEtaZeta = np.zeros([len(v),3]) 
        if psiVals is not None:
            self.psiEtaZeta[:,0] = psiVals
        if etaVals is not None:
            self.psiEtaZeta[:,1] = etaVals
        if zetaVals is not None:
            self.psiEtaZeta[:,2] = zetaVals

    def getParam(self):
        return self.psiEtaZeta

    def calcPsiEtaZeta(self, surface):
        transPts = self.calcTransformPoints(surface)
        self.extParam.coords=np.array(list(zip(transPts[:,1],np.zeros(len(transPts[:,1])))))
        etaVals = []
        psiVals = []
        for i in range(self.numPts):
            etaVals.append(self.extParam.calcXToPsi([transPts[i,1]])[0])
            leCoord = self.extParam.calcXToZ([transPts[i,1]])[0]*self.psiAx+transPts[i,1]*self.etaAx
            leCoordMod = self.extModFunc([leCoord],etaVals[-1],self.extModCoeffs)[0]
            self.csParam.refLen = self.chordModFunc(etaVals[-1],self.chordModCoeffs)
            psiVals.append(self.csParam.calcXToPsi([self.surface[i,0]-leCoordMod[0]])[0])
        zetaVals = self.calcZeta(np.array(psiVals), np.array(etaVals))
        return np.array(list(zip(psiVals,etaVals,list(zetaVals))))

    def calcZeta(self, psiVals, etaVals):
        zetaVals = []
        for i in range(len(psiVals)):
            leCoord = self.extParam.calcPsiToX([etaVals[i]])[0]*self.etaAx+self.extParam.calcZetaToZ([psiVals[i]])[0]*self.psiAx
            leCoordMod = self.extModFunc([leCoord],etaVals[i],self.extModCoeffs)[0]
            self.csParam.refLen = self.chordModFunc(etaVals[i],self.chordModCoeffs)
            csCoord = self.csParam.calcXToZ([psiVals[i]*self.csParam.refLen])[0]*self.zetaAx+self.csParam.calcPsiToX([psiVals[i]])[0]*self.psiAx+leCoordMod
            zeta = self.csParam.calcZToZeta([csCoord[2]-leCoordMod[2]])[0]
            zetaVals.append(zeta)
        return np.array(zetaVals)

    def updateZeta(self):
        self.psiEtaZeta[:,2] = self.calcZeta(self.psiEtaZeta[:,0],self.psiEtaZeta[:,1])

    def calcTransform(self,inverse=False):
        mat = np.array([np.transpose(self.psiAx), np.transpose(self.etaAx), np.transpose(self.zetaAx)])
        if inverse:
            mat = np.linalg.inv(mat)
        return mat

    def transformPoint(self,pt,inv=False):
        return np.dot(self.calcTransform(inv),pt)

    def calcTransformPoints(self, pts, inv=False):
        transformedPoints = []
        for pt in pts:
            transformedPoints.append(self.transformPoint(pt)-self.origin*(-1.0 if inv else 1.0))
        return np.array(transformedPoints)

    def updateTransformedPoints(self,surface):
        transformedPoints = []
        for pt in surface:
            transformedPoints.append(self.transformPoint(pt))
        self.transformedPoints = np.array(transformedPoints)

    def calcSurface(self,psiEtaZeta):
        leCoords = np.array([self.extParam.calcPsiToX(psiEtaZeta[:,1])[_]*self.etaAx for _ in range(len(psiEtaZeta))])
        leCoordMods = np.array([self.extModFunc([leCoords[_]], psiEtaZeta[_,1], self.extModCoeffs)[0] for _ in range(len(psiEtaZeta))])
        surface = []
        for i in range(len(psiEtaZeta)):
            self.csParam.refLen = self.chordModFunc(psiEtaZeta[i,1],self.chordModCoeffs)
            coord = (leCoordMods[i,:]+self.csParam.calcPsiToX([psiEtaZeta[i,0]])[0]*self.psiAx+
                     self.csParam.calcZetaToZ([psiEtaZeta[i,2]])[0]*self.zetaAx)
            surface.append(self.csModFunc([coord], psiEtaZeta[i,1], self.csModCoeffs)[0]+self.origin)
        return np.array(surface)

    def updateCoords(self):
        self.updateZeta()
        self.surface = self.calcSurface(self.psiEtaZeta)

    def getCoords(self):
        return self.surface

    def defaultExtrudeFunc(self, eta, *coeffs):
        return 0

    def defaultClassFunc(self, psi, *coeffs):
        return 0

    def defaultExtModFunc(self, pts, eta, *coeffs):#Modifies leading edge points
        newPts = []
        for p in pts:
            newPts.append(p)
        return np.array(newPts)

    def defaultCsModFunc(self, pts, eta, *coeffs):
        newPts = []
        for p in pts:
            newPts.append(p)
        return np.array(newPts)

    def defaultChordModFunc(self, eta, *coeffs):
        return 1.0

    def updateCoeffs(self, *coeffs):
        coeffs=coeffs[0]
        n1 = len(self.extShapeCoeffs)
        if len(coeffs)>n1 and n1>0:
            self.extShapeCoeffs = coeffs[:n1]
            self.extOffset = coeffs[n1]
        n2 = n1+self.order[0]+1
        if len(coeffs)>n2 and n2>n1:
            self.csShapeCoeffs = coeffs[n1+1:n2]
            self.csOffset = coeffs[n2]
        n3 = n2+len(self.extModCoeffs)
        if len(coeffs)>=n3 and n3>n2:
            self.extModCoeffs = coeffs[n2+1:n3]
        n4 = n3+len(self.csModCoeffs)
        if len(coeffs)>=n4 and n4>n3:
            self.csModCoeffs = coeffs[n3:n4]
        n5 = n4+len(self.chordModCoeffs)
        if len(coeffs)>=n5 and n5>n4:
            self.chordModCoeffs = coeffs[n4:n5]
        n6 = n5+len(self.extClassCoeffs)
        if len(coeffs)>=n6 and n6>n5:
            self.extClassCoeffs = coeffs[n5:n6]
        n7 = n6+len(self.csClassCoeffs)
        if len(coeffs)>=n7 and n7>n6:
            self.csClassCoeffs = coeffs[n6:n7]

    def getCoeffs(self):
        coeffs = []
        for _ in self.extShapeCoeffs:
            coeffs.append(_)
        coeffs.append(self.extOffset)
        for _ in self.csShapeCoeffs:
            coeffs.append(_)
        coeffs.append(self.csOffset)
        for _ in self.extModCoeffs:
            coeffs.append(_)
        for _ in self.csModCoeffs:
            coeffs.append(_)
        for _ in self.chordModCoeffs:
            coeffs.append(_)
        for _ in self.extClassCoeffs:
            coeffs.append(_)
        for _ in self.csClassCoeffs:
            coeffs.append(_)
        return coeffs

    def fit3d(self):
        def surface(XYVals,*coeffs):
            self.updateCoeffs(coeffs)
            numPts = len(XYVals)
            coords = np.array([self.transformPoint(np.append(XYVals[_],0)) for _ in range(numPts)])
            psiEtaZeta = self.calcPsiEtaZeta(coords)
            return self.calcSurface(psiEtaZeta)[2]

        #TODO Fix warning for cov params being inf in certain conditions
        inpCoeffs = self.extShapeCoeffs+[self.extOffset]+self.csShapeCoeffs+[self.csOffset]+self.extModCoeffs+self.csModCoeffs+self.chordModCoeffs+self.extClassCoeffs+self.csClassCoeffs
        coeffs,cov = scp.curve_fit(surface,self.surface[:,0:2],self.surface[:,2],inpCoeffs)
        self.updateCoeffs(coeffs)
        self.updatePsiEtaZeta()

class CSTFairing3D(CST3DParam):
    """
    Class for storing cst parameterization information for "fairings" (joins n curves with surface)
    Takes input of multiple boundaries which are to be connected by fairing surface

    Parameters
    -----
    boundaries : list(ndarray(nPts,3))
        List of prescribed boundaries, each boundary is an ndarray of coordinates on boundary curve

    growthAngleFunc : Function(psi, *coeffs)
        Function to describe offset to normal growth angle from boundaries
        returns - growthAngle, float radians

    growthAngleCoeffs : List(nBoundaries,List(coeffs))
        Coefficients to add to growth angle functions
    """

    def __init__(self, boundaries, growthAngleFunc=None, growthAngleCoeffs=[]):
        self.nBounds = len(boundaries)
        self.growthAngleFunc = [self.defaultGrowthAngleFunc for _ in range(self.nBounds)] if growthAngleFunc is None else growthAngleFunc
        self.growthAngleCoeffs = [[0] for _ in range(self.nBounds)] if len(growthAngleCoeffs)!=self.nBounds else growthAngleCoeffs
        
        self.closed = None
        self.boundaries = OrderedDict()
        self.setBoundaries(self.boundaries)

        self.centroid = np.array([0,0,0])
        self.sections = OrderedDict()
        self.setSections(self.boundaries)
        ##Compute everything needed to define intermediate surface
        self.surface = None
        

    def defaultGrowthAngleFunc(self, nu, *coeffs):
        #Normal offset angle along curve, will use local boundary vector for rotation
        coeffs = coeffs[0]
        return coeffs[0]

    def setBoundaries(self, boundaries):
        #Set boundary dict from boundaries list
        types = []
        self.nBounds = len(boundaries)
        self.boundaries = OrderedDict()
        for _ in range(self.nBounds):
            nPts = len(boundaries[_])
            curBoundary = 'boundary'+str(_+1)
            self.boundaries[curBoundary] = {}
            self.boundaries[curBoundary]['growthAngle'] = {'func': self.growthAngleFunc[_], 'coeffs': self.growthAngleCoeffs[_]}
            self.boundaries[curBoundary]['curve'] = Curve(boundaries[_])

            curve = self.boundaries[curBoundary]['curve']
            types.append(curve.closed)

        if np.all(types==True):
            self.closed = True
        elif np.all(types==False):
            self.closed = False
        else:
            raise Exception('Boundary type mismatch, Must be all closed or all open: ', 
                            ['boundary'+str(_+1)+' closed, '+str(types[_]) for _ in range(nPts)])

    def setNormals(self):
        for boundaryKey in self.boundaries.keys():
            boundary = self.boundaries[boundaryKey]
            curve = boundary['curve']
            newNormals = []
            for _ in range(curve.nPts):
                nu = curve.calcNuVal(_)
                rot = boundary['growthAngle']['func'](nu, boundary['growthAngle']['coeffs'])
                localVec = curve.pts[_+1]-curve.pts[_] if (_+1)<curve.nPts else curve.pts[-1]-curve.pts[-2]
                newNormals.append(rotation.rotVbyW(curve.normals[_],localVec,rot))
            curve.setNormals(np.array(newNormals))
            self.boundaries[boundaryKey]['curve'] = curve

    def setSections(self, boundaryDict):
        #Set section information from boundary dict
        self.sections = OrderedDict()
        boundaryKeys = boundaryDict.keys()
        self.nBounds = len(boundaryKeys)
        if self.nBounds <= 0:
            raise Exception("No boundaries in dictionary")

        #Compute centroids
        ptsList = [boundaryDict[key]['curve'].pts for key in boundaryKeys]
        self.centroid = Curve.calcCentroid(ptsList)
        curveCentroids = []
        for key in boundaryKeys:
            self.sections[key] = {}
            curve = boundaryDict[key]['curve']
            curveCentroids.append(curve.centroid)
            self.sections[key]['vertices'] = [self.centroid]

        #Compute vertices
        #For 1-3 boundaries, the sections are described by n (1-3) vertices, and it is projected in 2d
        #For larger, each prism consists of 4-5 points, self.centroid and other section vertices
        for _ in range(self.nBounds):
            if self.nBounds in [1]:
                pass
            elif self.nBounds in [2]:
                self.sections[boundaryKeys[_]]['vertices'].append(curveCentroids[_])
            elif self.nBounds in [3]:
                vertice = np.sum(curveCentroids[_-1]+curveCentroids[_])/2.0
                for i in range(2):
                    self.sections[boundaryKeys[_-i]]['vertices'].append(vertice)
            else:
                vertice = np.sum(curveCentroids[_-2]+curveCentroids[_-1]+curveCentroids[_])/3.0
                for i in range(3):
                    self.sections[boundaryKeys[_-i]]['vertices'].append(vertice)
        
        #Compute parameterization directions
        for _ in range(self.nBounds):
            boundaryKey = boundaryDict.keys()[_]
            if self.nBounds in [1]:
                #Nu ax assumed right hand ccw
                self.sections[boundaryKey]['localAxes'] = []
                pts = ptsList[_]
                for i in range(len(pts)):
                    localAxes = np.zeros([3,3])
                    if self.closed:
                        rVec = pts[i]-curveCentroids[_]
                        localAxes[0,:] = rVec/np.linalg.norm(rVec)
                        self.sections[boundaryKey]['localAxes'].append(localAxes)
                    else:
                        raise Exception('Cannot grow from singular unclosed curve')

    def calcSection(self, pt):
        #Calculate which section a point lies in
        boundaryKey = 'boundary0'
        return boundaryKey

    def calcSurface(self, boundaryDict):
        #Calculate surface coordinates
        for boundaryKey in boundaryDict.keys():
            pts = boundaryDict[boundaryKey]['curve']
            nPts = len(pts)
            locNorms = boundaryDict[boundaryKey]['localNorms']

        
class CSTWing3D(CST3DParam):
    """
    Class for storing cst parameterization information for wings
    """

    def __init__(self, surface, extrudeFunc=None, extClassCoeffs=[], extShapeCoeffs=[], extModFunc=None, extModCoeffs=[],
                 classFunc=None, csClassCoeffs=[0.5, 1.0], csShapeCoeffs=[], csModFunc=None, csModCoeffs=[], chordModFunc=None,
                 chordModCoeffs=[], origin=np.array([0.0,0.0,0.0]), refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), order=[7,2],
                 zScale=1.0):
        
        self.extrudeFunc = self.wingExtFunc if extrudeFunc is None else extrudeFunc
        self.extModFunc = self.wingExtModFunc if extModFunc is None else extModFunc
        self.classFunc = self.airfoilClassFunc if classFunc is None else classFunc
        self.csModFunc = self.airfoilModFunc if csModFunc is None else csModFunc
        self.chordModFunc = self.wingChordModFunc if chordModFunc is None else chordModFunc

        super().__init__(surface=surface, extrudeFunc=self.extrudeFunc, extClassCoeffs=extClassCoeffs, extShapeCoeffs=extShapeCoeffs, extModFunc=self.extModFunc,
                 extModCoeffs=extModCoeffs, classFunc=self.airfoilClassFunc, csClassCoeffs=csClassCoeffs, csShapeCoeffs=csShapeCoeffs, csModFunc=self.csModFunc,
                 csModCoeffs=csModCoeffs, chordModFunc=self.chordModFunc, chordModCoeffs=chordModCoeffs, origin=origin, refAxes=refAxes, order=order, zScale=zScale)
        
    def airfoilClassFunc(self, psi, *coeffs):
        powerTerms = coeffs[0]
        return np.power(psi,powerTerms[0])*np.power(1-psi,powerTerms[1])

    def airfoilModFunc(self, pts, eta, *coeffs):
        return super().defaultCsModFunc(pts, eta, coeffs)

    def wingExtFunc(self, eta, *coeffs):
        return super().defaultExtrudeFunc(eta, coeffs)

    def wingExtModFunc(self, pts, eta, *coeffs):
        return super().defaultExtModFunc(pts, eta, coeffs)

    def wingChordModFunc(self, eta, *coeffs):
        return super().defaultChordModFunc(eta, coeffs)


class CSTAirfoil3D(CST3DParam):
    """
    Class for storing cst parameterization information for extruded airfoil (Pseudo 2D)
    """
    def __init__(self, surface, csClassCoeffs=[0.5,1.0], csShapeCoeffs=[], origin=np.array([0.0,0.0,0.0]),
                 refAxes=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), order=[7,0], zScale=1.0):
        self.powerTerms = csClassCoeffs
        super().__init__(surface=surface, extrudeFunc=None, extClassCoeffs=[], extShapeCoeffs=[], extModFunc=None,
                 extModCoeffs=[], classFunc=self.airfoilClassFunc, csClassCoeffs=[], csShapeCoeffs=csShapeCoeffs, csModFunc=None,
                 csModCoeffs=[], chordModFunc=None, chordModCoeffs=[], origin=origin, refAxes=refAxes, order=order, zScale=zScale)

    def airfoilClassFunc(self, psi, *coeffs):
        return np.power(psi,self.powerTerms[0])*np.power(1-psi,self.powerTerms[1])




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

