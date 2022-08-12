from .autoFoam import *
from ..pyroc_utils import *
from argparse import ArgumentParser
import subprocess
import os

class SweepParser(ArgumentParser):
    #Class for running sweeps from terminal
    def __init__(self, description='Setup/Run Sweep'):
        self.add_argument('baseName', type=str, help='Base name for all created directories')
        self.add_argument('-d', '--rootDir', type=str, nargs='?', default=os.getcwd(), help='Case directory, must contain constant/polyMesh generated\'s')
        self.add_argument('-c', '--cases', type=str, nargs='?', default='cases.inp', help='Path to input Sweep Control File')
        self.add_argument('-t', '--turbModel', type=str, nargs='?', default='SpalartAllmaras', help='Turbulence Model to Use for All Runs: [SA, SAFV3, KE, SST]')
        self.add_argument('-n', '--nodes', type=int, default=1, help='Number of Nodes per job')
        self.add_argument('-r', metavar='START', dest='start', type=int, default=0, help='Start sweep on case START in the cases.inpt file')
        self.add_argument('-e', metavar='END', dest='end', type=int, default=0, help='End sweep on case END in the cases.inpt file')
        self.add_argument('-s', '--solver', type=str, nargs='?', default=None, help='Force solver instead of automatically determining from Mach')
        self.args = self.parse_args()
        pass

class SweepFOAM(object):
    #Class for running cfd sweeps

    def __init__(self, baseName, rootDir, cases='cases.inp', range=[0, None], forceSolver=None,
                 turbModel='SpalartAllmaras', wallFunctions=False, dirMode='flat', nNodes=1, ppn=36):
        self.NODES_PER_JOB = nNodes
        self.PROCS_PER_NODE = ppn
        self.queueType = 'None'
        self.runScriptName = 'runScript.py'

        self.baseName = baseName
        self.rootDir = rootDir
        
        #read input variables and cases
        casesPath = os.path.join(self.rootDir, cases) if cases=='cases.inp' else cases
        self.casesDict = self.readCases(casesPath, range[0], range[1])
        
        #Force the solver for all cases, otherwise choose from mach number
        self.forceSolver = forceSolver
        #Set turbulence model for all cases
        self.turbModel = turbModel
        #Set whether or not to use wall functions
        self.wallFunctions = wallFunctions
        #Directory splitting mode
        self.dirMode = dirMode

        #Setup shared options:
        self.sharedOptions = {
            'dirDict': {
                'rootDir': self.rootDir
            },
            #TODO Implement option to sweep turb models
            'turbulenceDict': {
                'model': self.turbModel,
                'wallFunctions': self.wallFunctions
            },
        }

        self.autoSweep = AutoFOAM(self.sharedOptions)
        #Read boundaries:
        boundaryFile = os.path.join(self.rootDir,'constant','polyMesh','boundary')
        self.boundaries = self.autoSweep.readBoundaries(boundaryFile)

        self.evalDict = {
            'CL': {
                'type': 'force',
                'directionMode': 'fixedDirection',
                'direction': 'liftDir',
            },
            'CD': {
                'type': 'force',
                'directionMode': 'fixedDirection',
                'direction': 'dragDir',
            },
            'CX': {
                'type': 'force',
                'directionMode': 'fixedDirection',
                'direction': [1,0,0]
            },
            'CY': {
                'type': 'force',
                'directionMode': 'fixedDirection',
                'direction': [0,1,0]
            },
            'CZ': {
                'type': 'force',
                'directionMode': 'fixedDirection',
                'direction': [0,0,1]
            },
            'CMX': {
                'type': 'moment',
                'directionMode': 'fixedDirection',
                'direction': [1,0,0]
            },
            'CMY': {
                'type': 'moment',
                'directionMode': 'fixedDirection',
                'direction': [0,1,0]
            },
            'CMZ': {
                'type': 'moment',
                'directionMode': 'fixedDirection',
                'direction': [1,0,0]
            },
        }

    def __call__(self):

        for caseName in self.casesDict.keys():
            case = self.casesDict[caseName]
            #sweepVars?

            if self.dirMode == 'flat':
                rootDir = os.path.join(self.rootDir, self.baseName)
            elif self.dirMode == 'tree':
                pass

            #Create case directory

    def readCases(self, casesPath, start=0, end=None):
        sweepDict = {}
        if (os.path.exists(casesPath)):
            file = open(casesPath, 'r')
            lines = file.readlines()
            self.inpVars = lines[0].split()
            for _ in self.inpVars:
                sweepDict[_] = np.array([])

            for _ in range(len(lines)-1)[start:end]:
                line = lines[_+1]
                splitLines = line.split()
                numVar = len(splitLines)
                if numVar>0 and line[0]!='#':
                    for __ in range(numVar):
                        sweepDict[self.inpVars[__]] = np.append(sweepDict[self.inpVars[__]], splitLines[__])
            file.close()
            return sweepDict
        else:
            raise Exception(casesPath + ' does not exist')

    def chooseSolver(self, M):
        solvers = ['DASimpleFoam', 'DARhoSimpleFoam', 'DARhoSimpleCFoam']
        if self.forceSolver is None:
            if M>0.99:
                print('Warning, supersonic solvers not well implemented')
            elif M>=0.5:
                solver = solvers[2]
            elif M>=0.1:
                solver = solvers[1]
            elif M>=0.01:
                solver = solvers[0]
            else:
                raise Exception('Mach Number Must Be Greater Than 0.01')
        elif self.forceSolver in solvers:
            solver = self.forceSolver
        else:
            raise Exception(self.forceSolver + ' is not handled')
        return solver

    def writeRunscript(self, file, solver, alpha0=None, CLStar=None):
        patches = [patch for patch in self.boundaries.keys() if self.boundaries[patch]['type']=='patch']
        walls = [wall for wall in self.boundaries.keys() if self.boundaries[wall]['type']=='wall']
        designSurfaces = str(walls)

        U0 = self.autoSweep.stateDict['U']
        rho0 = self.autoSweep.stateDict['rho']
        C0 = self.autoSweep.getOption('refDict')['refChord']
        A0 = self.autoSweep.getOption('refDict')['refArea']
        refLoc = str(self.autoSweep.getOption('refDict')['refLocation'])
        forceScale = str(1.0/(0.5*rho0*U0*U0*A0))
        momentScale = forceScale/C0

        #Imports
        file.write('import os\n')
        file.write('from mpi4py import MPI\n')
        file.write('from dafoam import PYDAFOAM, optFuncs\n')
        file.write('from pygeo import *\n')
        file.write('from idwarp import *\n')
        file.write('import numpy as np\n')
        file.write('\n')

        #TODO Remove necessity for this function by removing pygeo dependancy in pydaoptions cldriver
        if CLStar is not None:
            evalFuncs = ['CX','CY','CZ','CMX','CMY','CMZ']
            file.write('def driveCL(DASolver, CLStar, alpha0=1.0, relax=0.8, tol=1e-4, maxit=15):\n')
            file.write('    alpha = alpha0\n')
            file.write('    dAlpha = 1.0\n')
            file.write('    CL = 0\n')
            file.write('    output = {}\n')
            file.write('    evalFuncs = %s\n' % str(evalFuncs))
            file.write('    for i in range(maxit):\n')
            file.write('        inletU,dragDir,liftDir = alpha(aoa)\n')
            file.write('        DASolver.setOption(\'primalBC\', {\'U0\': {\'variable\': \'U\', \'patches\': %s, \'value\': inletU}})\n' % patches)
            file.write('        DASolver.updateDAOption()\n')
            file.write('        DASolver()\n')
            file.write('        DASolver.evalFunctions(output, evalFuncs=evalFuncs)\n')
            file.write('        aoa = np.radians(alpha)\n')
            file.write('        output[\'CL\'] = -output[\'CX\']*np.sin(aoa)+output[\'CZ\']*np.cos(aoa)\n')
            file.write('        output[\'CD\'] = output[\'CX\']*np.cos(aoa)+output[\'CZ\']*np.sin(aoa)\n')
            file.write('        if abs(CLStar-output[\'CL\'])/CLStar<tol:\n')
            file.write('            return aoa,output\n')
            file.write('        dCLdAlpha = 2*np.pi if i==0 else (output[\'CL\']-CL)/dAlpha\n')
            file.write('        dAlpha = relax*(CLStar-output[\'CL\'])/(dCLdAlpha)\n')
            file.write('        alpha += dAlpha\n')
            file.write('        CL = output[\'CL\']\n')
            file.write('        Info(\'\\nAngleOfAttack : \' + str(aoa) + \'\\n\')\n')
            file.write('    return aoa,CL,CD,output\n')
            file.write('\n')

        #Note: alpha currently set to XZ plane
        #TODO Generalize orientation
        if alpha0 is not None:
            file.write('def alpha(val):\n') #alpha in degrees
            file.write('    aoa = val * np.pi / 180.0\n')
            file.write('    inletU = [float(%f * np.cos(aoa)), 0.0, float(%f * np.sin(aoa))]\n' % (U0, U0))
            file.write('    dragDir = [float(np.cos(aoa)), 0.0, float(np.sin(aoa))]\n')
            file.write('    liftDir = [float(-np.sin(aoa)), 0.0, float(np.cos(aoa))]\n')
            file.write('    return inletU,dragDir,liftDir\n')
            file.write('\n')
            file.write('inletU,dragDir,liftDir=alpha(%f)\n' % alpha0)
            file.write('\n')
        
        file.write('daOptions = {\n')
        file.write('    \'designSurfaces\': %s,\n' % designSurfaces)
        file.write('    \'solverName\': \'%s\',\n' % solver)
        file.write('    \'primalMinResTol\': %f,\n' % self.resTol)
        file.write('    \'primalBC\': {\n')
        file.write('        \'useWallFunction\': %s,\n' % str(self.wallFunctions))
        file.write('    },\n')
        file.write('    \'primalVarBounds\': {\n')
        file.write('        \'UMax\': 1000.0,\n')
        file.write('        \'UMin\': -1000.0,\n')
        file.write('        \'pMax\': 500000.0,\n')
        file.write('        \'pMin\': 20000.0,\n')
        file.write('        \'eMax\': 500000.0,\n')
        file.write('        \'eMin\': 100000.0,\n')
        file.write('        \'rhoMax\': 5.0,\n')
        file.write('        \'rhoMin\': 0.2,\n')
        file.write('    },\n')

        file.write('    \'objFunc\': {\n')
        for func in evalFuncs:
            funcDict = self.evalDict[func]
            if funcDict['type']=='force':
                file.write('        \'%s\': {\n' % func)
                file.write('            \'part1\': {\n')
                file.write('                \'type\': \'%s\',\n' % funcDict['type'])
                file.write('                \'source\': \'patchToFace\',\n')
                file.write('                \'patches\': %s,\n' % walls)
                file.write('                \'directionMode\': \'%s\',\n' % funcDict['directionMode'])
                file.write('                \'direction\': %s,\n' % str(funcDict['direction']))
                file.write('                \'scale\': %f,\n' % forceScale)
                file.write('                \'addToAdjoint\': True,\n')
                file.write('            }\n')
                file.write('        },\n')
            elif funcDict['type']=='moment':
                file.write('        \'%s\': {\n' % func)
                file.write('            \'part1\': {\n')
                file.write('                \'type\': \'%s\',\n' % funcDict['type'])
                file.write('                \'source\': \'patchToFace\',\n')
                file.write('                \'patches\': %s,\n' % walls)
                file.write('                \'axis\': %s,\n' % str(funcDict['direction']))
                file.write('                 \'center\': %s,\n' % refLoc)
                file.write('                \'scale\': %f,\n' % momentScale)
                file.write('                \'addToAdjoint\': True,\n')
                file.write('            }\n')
                file.write('        },\n')
        file.write('    },\n')
        file.write('    \'checkMeshThreshold\': {\'maxAspectRatio\': %d},\n' % self.MAX_AR)
        file.write('\n')
        file.write('DASolver = PYDAFOAM(options=daOptions, comm=MPI.COMM_WORLD)\n')
        file.write('\n')

        if CLStar is not None:
            file.write('alpha,output=constCL(%f,DASolver)\n' % CLStar)
        else:
            file.write('DASolver()\n')
            file.write('funcs = {}\n')
            file.write('evalFuncs = %s\n' % str(evalFuncs))
            file.write('DASolver.evalFunctions(output, evalFuncs)\n')

        file.close()

    def writeJobscript(self, file, dependID=None, dependCasePath=None, jobDir='.', wallTime='0:30:00', jobName='sweep_run', logFile='runLog.txt'):
        #Write jobsript for slurm queues on clusters
        if self.queueType=='slurm':
            file.write('#!/bin/bash\n')
            file.write('\n')
            if dependID is not None:
                file.write('#SBATCH --dependancy=afterok:%d' % dependID)
            file.write('#SBATCH --chdir=%s\n' % jobDir)
            file.write('#SBATCH --time=%s\n' % wallTime)
            file.write('#SBATCH --nodes=%d\n' % self.NODES_PER_JOB)
            file.write('#SBATCH --ntasks-per-node=%d\n' % self.PROCS_PER_NODE)
            file.write('#SBATCH --job-name=\"%s\"\n' % jobName)
            file.write('#SBATCH --output=\"%s\"\n' % logFile)
            file.write('\n')
            file.write('module load intel/18.2\n')
        elif self.queueType=='None':
            file.write('cd %s\n' % jobDir)
        else:
            raise Exception('Queue type not supported')

        file.write('source %s\n' % self.loadPath)
        #Restart from a solved flow
        if dependCasePath is not None:
            file.write('latestTime=$(foamListTimes -case %s -latestTime | tail -n 1)\n' % dependCasePath)
            file.write('cp -r %s/$latestTime 0\n' % dependCasePath)
            file.write('rm %s\n' % os.path.join('0', 'uniform', 'time'))
            file.write('changeDictionary -latestTime\n\n')

        file.write('mpirun -np %d python %s 2>&1 | tee %s\n' % (self.NODES_PER_JOB*self.PROCS_PER_NODE, self.runScriptName, logFile))
        file.write('reconstructPar\n')
        file.write('rm -r processor*')