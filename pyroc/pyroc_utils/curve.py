import numpy as np
import scipy.linalg as scp
import rotation
import matplotlib.pyplot as plt

class Curve():
    def __init__(self, pts, ordered=False):
        self.pts = pts if len(pts)>1 else None
        self.pts = self.deleteDuplicates(self.pts) if len(self.pts)>1 else self.pts
        self.pts = self.orderPoints(self.pts) if not ordered else self.pts
        self.nPts = len(self.pts)
        self.simple = self.checkSimple(self.pts) if self.nPts>1 else None
        self.closed = self.checkClosed(self.pts) if self.nPts>1 else None
        self.bounds = self.calcBoundingBox(self.pts) if self.nPts>1 else None
        self.length = self.calcCurveLength(self.pts) if self.nPts>1 else None
        self.normals = self.calcNormals(self.pts) if self.nPts>1 else None
        self.centroid = self.calcCentroid([self.pts]) if self.nPts>0 else None

    def orderPoints(self, points, closestPtInd=0):
        pts = list(points)
        nPts = len(pts)
        orderedPts = [pts.pop(0)]
        for i in range(nPts-1): 
            shortest=1e12
            for j in range(nPts-i-1):
                distance = np.linalg.norm(pts[j]-orderedPts[-1])
                if j==0 or distance<=shortest:
                    shortest = distance
                    closestPtInd = j
            orderedPts.append(pts.pop(closestPtInd))
        return np.array(orderedPts)

    def deleteDuplicates(self, points, tol=1e-15):
        nPts = len(points)
        new_pts = points[:]
        i, j = 0, 0
        while i < nPts-1:
            while j < nPts-1:
                if np.all(np.abs(new_pts[i]-new_pts[j])<tol) and i!=j:
                    new_pts = np.delete(new_pts,j,0)
                    nPts -= 1
                    i += 1
                    j = 0
                else:
                    j += 1
            i += 1
        return new_pts
        
    def calcBoundingBox(self, pts):
        bounds = np.array([[np.min(pts[:,0]), np.max(pts[:,0])],
                           [np.min(pts[:,1]), np.max(pts[:,1])],
                           [np.min(pts[:,2]), np.max(pts[:,2])]])
        return bounds

    def checkClosed(self, pts, spacingTol=1.1, smoothTol=120):
        if len(pts)<=2:
            return False

        first = pts[1]-pts[0]
        last = pts[0]-pts[-1]
        firstDistance = np.linalg.norm(first)
        maxDistance = np.max([np.linalg.norm(pts[_+1]-pts[_]) for _ in range(len(pts)-1)])
        lastDistance = np.linalg.norm(last)
        angle = np.arccos(np.dot(last,first)/(firstDistance*lastDistance)*(1-1e-16))

        if angle<abs(np.radians(smoothTol)) and lastDistance/maxDistance<spacingTol:
            return True
        else:
            return False

    def calcCurveLength(self, pts):
        #estimate length of ordered curve
        curveLen = 0
        nPts = len(pts)
        for i in range(nPts if self.closed else nPts-1):
            curveLen += np.linalg.norm(pts[(i+1)%nPts]-pts[i%nPts])
        return curveLen
    
    def getCurveLength(self):
        return self.calcCurveLength(self.pts)

    def checkSimple(self, pts, tol=1e-8):
        nPts = len(pts)
        if nPts<=2:
            return True
        #Check for intersection between line segments:
        closed = self.checkClosed(pts)
        nSeg = nPts if closed else nPts-1

        lineSegments = [LineSegment(pts[i],pts[(i+1)%nPts]) for i in range(nSeg)]
        i = 0
        while len(lineSegments)>1:
            l1 = lineSegments[0]
            if i+1 < len(lineSegments):
                l2 = lineSegments[1+i]
                i+=1
            else:
                i = 0
                lineSegments.pop(0)
            bDir = (l2.pts[0]-l1.pts[0])/np.linalg.norm(l2.pts[0]-l1.pts[0])
            colinear = (abs(abs(np.dot(l1.dir,l2.dir))-1.0) < tol) and (abs(abs(np.dot(l1.dir,bDir))-1.0) < tol)
            planeDirs = [np.cross(l1.dir,l2.dir),np.cross(l1.dir,bDir)]
            coplanar = np.all(np.cross(planeDirs[0],planeDirs[1])==0)

            #Intersection, same start or end
            if np.all(l1.pts[0]==l2.pts[0]) or np.all(l1.pts[1]==l2.pts[1]):
                return False

            elif colinear:
                s1 = l1.reverseInterpolate(l2.pts[0])
                s2 = l1.reverseInterpolate(l2.pts[1])
                if (s1<0 and s2<=0) or (s1>=1 and s2>=1):
                    pass
                else:
                    return False

            elif coplanar:
                isecMatA = np.array([[l1.a[_], -l2.a[_]] for _ in range(3)])
                isecMatB = l1.pts[1]-l2.pts[1]
                sol,reg,rank,eig = np.linalg.lstsq(isecMatA,isecMatB,rcond=None)
                s, t = sol[0], sol[1]
                if s>tol and s<=1 and t>=0 and t<1:
                    if np.all(np.dot(isecMatA,sol)-isecMatB<tol):
                        return False
            else:
                pass
            #i += 1
        return True

    def calcNormals(self, pts, tol=1e-15):
        nPts = len(pts)
        localNormals = []
        for i in range(nPts):
            vec1 = (pts[(i+1)%nPts]-pts[(i)%nPts])
            vec2 = (pts[(i+2)%nPts]-pts[(i+1)%nPts])
            normVec = np.cross(vec1,vec2)
            if np.linalg.norm(normVec)<tol:
                sub = np.array([1.0,0.0,0.0])
                if len(localNormals)>0:
                    normVec = localNormals[-1]
                elif np.any(vec1 != sub):
                    normVec = np.cross(sub,vec1)/np.linalg.norm(np.cross(sub,vec1))
                else:
                    normVec = np.array([0.0,1.0,0.0])
            else:
                normVec = normVec/np.linalg.norm(normVec)
            localNormals.append(normVec)
        return np.array(localNormals)

    def calcCentroid(self, ptsList):
        #Estimate center of curve
        weightedPts = []
        curveLengths = []
        for pts in ptsList:
            nPts = len(pts)
            for i in range(nPts-1):
                weightedPts.append(0.5*np.linalg.norm(pts[i%nPts+1]-pts[i%nPts])*(pts[i%nPts+1]+pts[i%nPts]))
            curveLengths.append(self.calcCurveLength(pts))
        return np.array(sum(weightedPts)/sum(curveLengths))

    def setNormals(self, normals):
        self.normals = normals

    def calcNuVal(self, index):
        curveLen = self.getCurveLength()
        indexLen = self.calcCurveLength(self.pts[:index+1])
        return indexLen/curveLen


class LineSegment():
    def __init__(self, pt1, pt2):
        # y=f(x)=(pt2-pt1)+pt1*x=b+ax 0<=x<=1
        self.pts = [pt1,pt2]
        self.b = pt1 #origin
        self.a = pt2-pt1 #segment vector
        self.length = np.linalg.norm(self.a)
        self.dir = self.a/self.length

    def interpolate(self, t):
        x = self.b+self.a*t
        return x

    def reverseInterpolate(self, x, tol=1e-15):
        t = None
        i = 0
        if (np.linalg.norm(x-self.b) == 0):
            return 0.0
        elif np.any(self.a != 0):
            dot = np.dot((x-self.b)/np.linalg.norm(x-self.b),self.dir)
        else:
            dot = 0
        if np.all([abs(abs(dot)-1.0) > tol for _ in range(3)]):
            return None
        while t is None:
            if self.a[i] != 0:
                t = (x[i]-self.b[i])/self.a[i]
            elif i==2:
                if np.all(x==self.b):
                    return 0.0
                else:
                    return None
            else:
                i+=1
        return t
        
#Test
# r = 1.0        
# t = np.linspace(0,2*np.pi,36)
# pts = np.array([[r*np.cos(t), r*np.sin(t), 0] for t in t[:-2]])
# curve = Curve(pts)
# print(curve.closed)
# print(curve.simple)
# print(curve.length)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot3D(curve.pts[:,0],curve.pts[:,1],curve.pts[:,2])
# ax.scatter3D(curve.centroid[0],curve.centroid[1],curve.centroid[2])
# for _ in range(curve.nPts):
#     ax.quiver(curve.pts[_,0],curve.pts[_,1],curve.pts[_,2], curve.normals[_,0], curve.normals[_,1], curve.normals[_,2], length=1)
# plt.show()