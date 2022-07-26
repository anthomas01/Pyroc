from .autoFoam import *
from ..pyroc_utils import *
import subprocess
import os

#TODO implement gathering data on multiple sections with different reference values

class SweepFOAM(object):
    #Class for running cfd sweeps

    def __init__(self, baseName, rootDir, cases='cases.inp', rrange=[0, None], forceSolver=None,
                 turbModel='SpalartAllmaras', wallFunctions=False, dirMode='flat', nNodes=1, ppn=36,
                 queueType='None', writeParaview=True, refChord=1.0, refArea=1.0, refLoc=[0.25,0.0,0.0],
                 dryRun=False):
        #Job Parameters
        self.NODES_PER_JOB = nNodes
        self.PROCS_PER_NODE = ppn
        self.queueType = queueType
        self.dryRun = dryRun

        #Path constants
        self.loadEnv = '/home/andrew/bin/loadDAFoam.sh' #TODO
        self.runScriptName = 'runScript.py'
        self.jobScriptName = 'foamJob'
        self.logName = 'log.txt'
        self.baseName = baseName
        self.rootDir = rootDir
        self.writeParaview = writeParaview

        self.resTol = 1e-6 #TODO
        self.maxAR = 1000000
        
        #read input variables and cases
        if type(cases)==dict:
            self.casesDict = cases
        else:
            casesPath = os.path.join(self.rootDir, cases) if cases=='cases.inp' else cases
            self.casesDict = self.readCases(casesPath, rrange[0], rrange[1])
        
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
                'rootDir': self.rootDir,
            },
            #TODO Implement option to sweep turb models
            'turbulenceDict': {
                'model': self.turbModel,
                'wallFunctions': self.wallFunctions,
            },
            'refDict': {
                'refChord': refChord,
                'refArea': refArea,
                'refLocation': refLoc,
            }
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
        caseKeys = list(self.casesDict.keys())
        for _ in range(len(caseKeys)):
            case = self.casesDict[caseKeys[_]]
            if (_==0) and ('RESTART' in self.inpVars) and (case['RESTART'] not in [0]):
                raise Exception('First case cannot have a restart code of %d' % case['RESTART'])

            #Create case name
            if self.dirMode == 'flat':
                caseList = [self.baseName]
                for var in self.sweptVars:
                    caseList.append(var+str(case[var]))
                caseName = '_'.join(caseList)
            elif self.dirMode == 'tree':
                caseName = self.baseName
                for var in self.sweptVars:
                    caseName = os.path.join(caseName, var+str(case[var]))
            else:
                raise Exception('Directory creation mode %s not implemented' % self.dirMode)
            caseDir = os.path.join(self.rootDir, caseName)
            self.casesDict[caseKeys[_]]['PATH'] = caseDir

            if self.dryRun:
                print(caseDir)
            elif 'RESTART' in self.inpVars and case['RESTART'] in [1]:
                #Write change dictionary for controlDict
                pass
            elif 'RESTART' in self.inpVars and case['RESTART'] in [3]:
                #Set case info from matching case
                pass
            else:
                #Make directory
                if not os.path.exists(caseDir):
                    os.mkdir(caseDir)

                #Set options in AutoFOAM
                self.autoSweep.setOption('dirDict', {'rootDir': caseDir})
                if 'RESTART' in self.inpVars:
                    self.autoSweep.setOption('controlDict', {'restart': case['RESTART']})
                self.autoSweep.setOption('controlDict', {'numIter': case['NUMITER'], 'printInterval': case['NUMITER']})
                self.autoSweep.solver = self.chooseSolver(case['MACH'])
                solverVars = self.autoSweep.getOption('solverDict')[self.autoSweep.solver]

                #Update flow vars
                for varName in case.keys():
                    self.autoSweep.stateDict[varName] = case[varName]
                self.autoSweep.updateStateDict()

                #Make system files
                self.autoSweep.writeControlDict()
                self.autoSweep.writeFvSchemesDict()
                self.autoSweep.writeFvSolutionDict()
                self.autoSweep.writeDecomposeParDict(self.PROCS_PER_NODE*self.NODES_PER_JOB)
                if 'RESTART' in self.inpVars and case['RESTART'] in [2]:
                    self.autoSweep.writeChangeDictionaryDict(solverVars)

                #Make constant files
                self.autoSweep.writeTurbulenceProperties()
                if self.autoSweep.solver in ['DASimpleFoam']:
                    self.autoSweep.writeTransportProperties()
                else:
                    self.autoSweep.writeThermophysicalProperties()

                #Copy Mesh to constant
                dirDict = self.autoSweep.getOption('dirDict')
                oldPolyDir = os.path.join(self.rootDir, dirDict['constDir'], dirDict['meshDir'])
                newPolyDir = os.path.join(caseDir, dirDict['constDir'], dirDict['meshDir'])
                copyDir(oldPolyDir, newPolyDir)

                #Make 0 files
                for var in solverVars:
                    self.autoSweep.writeBC(var)
                
                #Make miscellaneous files
                if self.writeParaview:
                    self.autoSweep.writeParaview()

                #Write runscript
                runScriptPath = os.path.join(caseDir, self.runScriptName)
                runScriptFile = open(runScriptPath,'w')
                alpha = None
                clStar = None
                if 'ALPHA' in self.inpVars:
                    alpha = case['ALPHA']
                else:
                    alpha = 0.0
                if 'CL' in self.inpVars:
                    clStar = case['CL']
                elif 'ALPHA' not in self.inpVars:
                    raise Exception('Needs either alpha or cl')
                self.writeRunscript(runScriptFile, self.autoSweep.solver, alpha0=alpha, CLStar=clStar)
                runScriptFile.close()

                #Write jobscript
                self.casesDict[caseKeys[_]]['dependID'] = None
                self.casesDict[caseKeys[_]]['dependPath'] = None
                if 'RESTART' in self.inpVars and case['RESTART'] in [2]:
                    self.casesDict[caseKeys[_]]['dependID'] = self.casesDict[caseKeys[_-1]]['ID']
                    self.casesDict[caseKeys[_]]['dependPath'] = self.casesDict[caseKeys[_-1]]['PATH']
                
                if self.queueType.upper() in ['NONE']:
                    jobScriptName = self.jobScriptName + '.sh'
                    jobScriptPath = os.path.join(caseDir, jobScriptName)
                    jobScriptFile = open(jobScriptPath, 'w')
                    self.writeJobScript(jobScriptFile, dependID=self.casesDict[caseKeys[_]]['dependID'], dependCasePath=self.casesDict[caseKeys[_]]['dependPath'],
                                        jobDir=caseDir, wallTime='0:30:00', jobName='foam_sweep', logFile=self.logName)
                    jobScriptFile.close()
                elif self.queueType.upper() in ['SLURM']:
                    jobScriptName = self.jobScriptName + '.sh'
                    jobScriptPath = os.path.join(caseDir, jobScriptName)
                    jobScriptFile = open(jobScriptPath, 'w')
                    self.writeJobScript(jobScriptFile, dependID=self.casesDict[caseKeys[_]]['dependID'], dependCasePath=self.casesDict[caseKeys[_]]['dependPath'],
                                        jobDir=caseDir, wallTime='0:30:00', jobName='foam_sweep', logFile=self.logName)
                    jobScriptFile.close()
                elif self.queueType.upper() in ['CONDOR']:
                    jobScriptName = self.jobScriptName + '.sh'
                    jobScriptPath = os.path.join(caseDir, jobScriptName)
                    jobScriptFile = open(jobScriptPath, 'w')
                    self.writeJobScript(jobScriptFile, dependID=self.casesDict[caseKeys[_]]['dependID'], dependCasePath=self.casesDict[caseKeys[_]]['dependPath'],
                                        jobDir=caseDir, wallTime='0:30:00', jobName='foam_sweep', logFile=self.logName)
                    jobScriptFile.close()

                #Adjust file permissions
                permissions = ['chmod', '-R', '777', caseDir]
                returnSubprocess(permissions)

                #Submit Job
                if self.queueType.upper() in ['NONE']:
                    jobid = _
                    self.casesDict[caseKeys[_]]['ID'] = jobid
                elif self.queueType.upper() in ['SLURM']:
                    cmd = ['sbatch', jobScriptPath]
                    #jobid = returnSubprocess(cmd)#TODO Get job id properly
                    jobid = _
                    self.casesDict[caseKeys[_]]['ID'] = jobid
                elif self.queueType.upper() in ['CONDOR']:
                    jobid = _
                    self.casesDict[caseKeys[_]]['ID'] = jobid
                    subScriptName = self.jobScriptName + '.sub'
                    subScriptPath = os.path.join(caseDir, subScriptName)
                    subScriptFile = open(subScriptPath, 'w')
                    self.writeCondorSub(subScriptFile)
                    subScriptFile.close()
                else:
                    raise Exception('Queue type %s not supported' % self.queueType)
            
            if self.queueType in ['condor']:
                dagScriptName = self.jobScriptName + '.dag'
                dagScriptPath = os.path.join(self.rootDir, dagScriptName)
                dagScriptFile = open(dagScriptPath, 'w')
                self.writeCondorDAG(dagScriptFile)
                dagScriptFile.close()
                cmd = ['condor_submit_dag', dagScriptName]
                #returnSubprocess(cmd)
            

    def readCases(self, casesPath, start=0, end=None):
        casesDict = OrderedDict()
        self.inpVars = []
        self.sweptVars = []

        if (os.path.exists(casesPath)):
            file = open(casesPath, 'r')
            lines = file.readlines()
            self.inpVars = [_.upper() for _ in lines[0].split()]

            for _ in range(len(lines)-1)[start:end]:
                line = lines[_+1]
                splitLines = line.split()
                numVar = len(splitLines)
                if numVar==len(self.inpVars) and line[0]!='#':
                    caseDict = {}
                    for __ in range(numVar):
                        caseDict[self.inpVars[__]] = float(splitLines[__])
                    casesDict['case%d' % (_+1)] = caseDict

                    #Check for changing vars
                    if _>0:
                        if ((casesDict['case%d' % (_+1)][self.inpVars[__]] != splitLines[__])
                             and (self.inpVars[__] not in self.sweptVars)):
                            if self.inpVars[__] not in ['RESTART']:
                                self.sweptVars.append(self.inpVars[__])
                        
            file.close()
            return casesDict
        else:
            raise Exception(casesPath + ' does not exist')

    def chooseSolver(self, M):
        solvers = ['DASimpleFoam', 'DARhoSimpleFoam', 'DARhoSimpleCFoam']
        if self.forceSolver is None:
            if M>0.99:
                print('Warning, supersonic solver not implemented')
            elif M>=0.5:
                solver = solvers[2]
            elif M>=0.1:
                solver = solvers[1]
            elif M>0:
                solver = solvers[0]
            else:
                raise Exception('Mach number must be greater than 0')
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
        rho0 = self.autoSweep.stateDict['RHO']
        C0 = self.autoSweep.getOption('refDict')['refChord']
        A0 = self.autoSweep.getOption('refDict')['refArea']
        refLoc = str(self.autoSweep.getOption('refDict')['refLocation'])
        forceScale = 1.0/(0.5*rho0*U0*U0*A0)
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
            file.write('def driveCL(DASolver, CLStar, alpha0=%f, relax=0.8, tol=1e-4, maxit=15):\n' % alpha0)
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
            file.write('            return alpha,output\n')
            file.write('        dCLdAlpha = 2*np.pi if i==0 else (output[\'CL\']-CL)/dAlpha\n')
            file.write('        dAlpha = relax*(CLStar-output[\'CL\'])/(dCLdAlpha)\n')
            file.write('        alpha += dAlpha\n')
            file.write('        Info(\'\\AngleOfAttack : \' + str(alpha) + \'\\n\')\n')
            file.write('    return alpha,output\n')
            file.write('\n')
        elif alpha0 is not None:
            evalFuncs = ['CD','CL','CMX','CMY','CMZ']
            #Note: alpha currently set to XZ plane
            #TODO Generalize orientation
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
        file.write('    \'checkMeshThreshold\': {\'maxAspectRatio\': %d},\n' % self.maxAR)
        file.write('}\n')
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

    def writeJobScript(self, file, dependID=None, dependCasePath=None, jobDir='.', wallTime='0:30:00', jobName='sweep_run', logFile='foamLog.txt'):
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

        elif self.queueType.upper() in ['NONE', 'condor']:
            file.write('#!/bin/bash\n')
            file.write('\n')
            file.write('cd %s\n' % jobDir)
        else:
            raise Exception('Queue type %s not supported' % self.queueType)

        file.write('source %s\n' % self.loadEnv)
        #Restart from a solved flow
        if dependCasePath is not None:
            file.write('latestTime=$(foamListTimes -case %s -latestTime | tail -n 1)\n' % dependCasePath)
            file.write('cp -r %s/$latestTime 0\n' % dependCasePath)
            file.write('rm %s\n' % os.path.join('0', 'uniform', 'time'))
            file.write('changeDictionary -latestTime\n\n')

        file.write('mpirun -np %d python %s 2>&1 | tee %s\n' % (self.NODES_PER_JOB*self.PROCS_PER_NODE, self.runScriptName, logFile))
        file.write('reconstructPar\n')
        file.write('rm -r processor*')

    def writeCondorSub(self, file, logFile='foamLog.txt'):
        file.write('# file name: %s.sub\n' % self.jobScriptName)
        file.write('Universe     = vanilla\n')
        file.write('Executable   = %s.sh\n' % self.jobScriptName)
        file.write('Requirements = Cpus >= %d && (PartitionableSlot || Cpus <= 8)\n' % self.PROCS_PER_NODE)
        file.write('Request_cpus = Cpus > %d ? %d : Cpus\n' % (self.PROCS_PER_NODE, self.PROCS_PER_NODE))
        file.write('Log          = condor.log\n')
        file.write('Output       = %s\n' % logFile)
        file.write('Error        = condor.error\n')
        file.write('Queue')

    def writeCondorDAG(self, file):
        file.write('# file name: %s.dag\n' % self.jobScriptName)
        caseKeys = list(self.casesDict.keys())
        for _ in range(len(caseKeys)):
            case = self.casesDict[caseKeys[_]]
            jobName = caseKeys[_]
            jobPath = os.path.join([case['PATH'], self.jobScriptName+'.sub'])
            file.write('Job %s %s\n' % (jobName, jobPath))
        for _ in range(len(caseKeys)):
            case = self.casesDict[caseKeys[_]]
            remainingKeys = np.delete(caseKeys,_)
            children = []
            for __ in range(len(remainingKeys)):
                compCase = self.casesDict[remainingKeys[__]]
                if compCase['dependID']==case['ID']:
                    children.append(remainingKeys[__])
            if len(children)>0:
                file.write('Parent %s Child %s\n' % (caseKeys[_], ' '.join(children)))
