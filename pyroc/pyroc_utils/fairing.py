from collections import OrderedDict
from .cst3d import *
from .curve import *
from .rotation import *

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
                newNormals.append(rotVbyW(curve.normals[_],localVec,rot))
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