import sys
import os
import re
import argparse
import subprocess
import numpy as np

#Arguments
parser = argparse.ArgumentParser(description='Run OpenFoam Sweep')
parser.add_argument('baseName', type=str, help='Base name for all created directories')
parser.add_argument('-d', '--rootdir', type=str, nargs='?', default='.', help='Case Directory Including Generated Mesh and BC\'s')
parser.add_argument('-c', '--ctlfile', type=str, nargs='?', default='./sweep.ctl', help='Input Sweep Control File')
parser.add_argument('-t', '--turbModel', type=str, nargs='?', default='SA', help='Turbulence Model to Use for All Runs: [SA, SAFV3, KE, SST]')
parser.add_argument('-n', '--nodes', type=int, default=1, help='Number of Nodes')
parser.add_argument('-q', '--queue', default='default', help='submit jobs to the specified queue (default = "default")')
parser.add_argument('-r', metavar='START', dest='start', type=int, default=0, help='start sweep on case START in the cases.inpt file')
parser.add_argument('-e', metavar='END', dest='end', type=int, default=0, help='end sweep on case END in the cases.inpt file')
#self.add_argument('-s','--starred',action='store_true', help='only run cases with a * in the first character')
args = parser.parse_args()

#Sample Input File (sweep.ctl)
#Implemented Options are: [ALPHA MACH RE TEMP] [NUMITER RESTART]
#                         [CL MACH RE TEMP]
'''
ALPHA MACH RE      TEMP NUMITER RESTART
3     0.5  6000000 273  2000    0
3     0.5  8000000 273  2000    2
'''

#Constants
GAMMA = 1.4 #Heat Capacity Ratio
R = 8.3144626 #Universal Gas Constant (J/mol*K)
D = 1 #Chord Length (m)
PRT = 0.85 #Turbulent Prandtl Number
PR = 0.72 #Prandtl Number
TVR = 10 #Turbulent Viscosity Ratio
TIR = 0.05 #Turbulent Intensity Ratio
MOLW = 28.964 #Gas Molecular Weight (g/mol)
RGAS = R*1e3/MOLW #Air Gas Constant
HFORM = 0
MAX_AR = 20000 #Maximum Mesh Aspect Ratio
A0 = D*1 #Reference Area
PRIMAL_MIN_RES_TOL = '1.0e-6'
WALL_FUNCTION = 'True'
runScrName = 'runScript.py'
ppn = int(os.getenv('PROCS_PER_NODE'))

solvers = ['DASimpleFoam','DARhoSimpleFoam','DARhoSimpleCFoam']
turbulenceModels = {
    'SA': 'SpalartAllmaras',
    'SAFV3': 'SpalartAllmarasFv3',
    'KE': 'kEpsilon',
    'SST': 'kOmegaSST'}
    

#Global Variables
numRuns = 0
inpVars = np.array([])

#Functions
def copyDir(old,new):
    if os.name=='nt':
        dirCopy = 'Xcopy /e /i /y '
        os.system(dirCopy + old.replace('/','\\') + ' ' + new.replace('/','\\'))
    else:
        dirCopy = 'cp -r '
        os.system(dirCopy + old + ' ' + new)
    return 0
    
def copyFile(old,new):
    if os.name=='nt':
        fileCopy = 'copy /v /y '
        os.system(fileCopy + old.replace('/','\\') + ' ' + new.replace('/','\\'))
    else:
        fileCopy = 'cp '
        os.system(fileCopy + old + ' ' + new)
    return 0
    
def delDir(rm):
    if os.name=='nt':
        dirDel = 'rmdir /s /q '
        os.system(dirDel + rm.replace('/','\\'))
    else:
        dirDel = 'rm -r '
        os.system(dirDel + rm)
    return 0
    
def delFile(rm):
    if os.name=='nt':
        fileDel = 'del /f /q '
        os.system(fileDel + rm.replace('/','\\'))
    else:
        fileDel = 'rm '
        os.system(fileDel + rm)
    return 0

def sutherland(T):
    MU0 = 1.716e-5
    T0 = 273.15
    C = 110.4
    MU = MU0*np.power(T/T0,1.5)*((T0+C)/(T+C))
    return MU

def perfectGas(rho,var,method):
    if method == 'T':
        T = var/(rho*RGAS)
        return T
    elif method =='P':
        P = var*rho*RGAS
        return P
    else:
        raise Exception('Perfect gas method must be \'T\' or \'P\'')

def mach(Ma,T):
    u = Ma*np.sqrt(GAMMA*RGAS*T)
    return u

def reynoldsNum(reynolds,mu,vel,length):
    rho = reynolds*mu/(vel*length)
    return rho

def turb(nu,U):
    nu_t = nu*TVR
    alpha_t = nu_t/PRT
    nuTilda = 5*nu
    for i in range(100):
        x3 = np.power(nuTilda/nu,3)
        nuTilda = nu_t*(x3+np.power(7.1,3))/x3
           
    k = 1.5*np.power(U*TIR,2)
    omega = k/nu_t
    epsilon = 0.09*np.power(k,2)/nu_t

    return nu_t, alpha_t, nuTilda, k, omega, epsilon
    
def getUVec(vel,alpha):
    return [float(vel * np.cos(alpha * np.pi / 180)), 0.0, float(vel * np.sin(alpha * np.pi / 180))]

def chooseSolver(M):
    if M>0.99:
        raise Exception('Mach Number Must Be Less Than 0.99')
    elif M>=0.6:
        solver = solvers[2]
    elif M>=0.1:
        solver = solvers[1]
    elif M>=0.01:
        solver = solvers[0]
    else:
        raise Exception('Mach Number Must Be Greater Than 0.01')
    return solver

def getFloats(line):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    out = rx.findall(line)
    return np.array([float(i) for i in out])

def getBC():
    boundaryFile = args.rootdir + '/constant/polyMesh/boundary'
    if (os.path.exists(boundaryFile)):
        f = open(boundaryFile, 'r')
        lines = f.readlines()

        begin = True
        patchName = False
        numPatch = 0
        BC = np.array([])
        for i in range(len(lines)):
            line = lines[i]
            splitLine = line.split()
            if len(splitLine) > 0:
                if splitLine[0]=='(' and begin:
                    begin = False
                    patchName = True

                elif not begin:
                    if patchName:
                        name = splitLine[0]
                        patchName = False
                    elif splitLine[0]=='type':
                        if len(BC) == 0:
                            BC = np.array([name,splitLine[1][:-1]])
                        else:
                            BC = np.vstack([BC,np.array([name,splitLine[1][:-1]])])
                        numPatch += 1
                    elif splitLine[0]=='}':
                        patchName = True
        f.close()
        return BC
    else:
        raise Exception(boundaryFile + ' does not exist')

def readCtl(collect=0):
    ctlfile = args.ctlfile
    sweepDict = {'RESTART' : 0, 'NUMITER' : 1000}
    global numRuns
    global inpVars

    if (os.path.exists(ctlfile)):
        f = open(ctlfile, 'r')
        lines = f.readlines()
        
        for i in range(len(lines)):
            line = lines[i]
            splitLines = line.split()
            
            if len(splitLines)>0 and line[0]!='#':
                if collect==0:
                    inpVars = np.array(splitLines)
                    for j in inpVars:
                        sweepDict[j] = np.zeros([1])
                    collect = 1
                        
                else:
                    readData = getFloats(line)
                    for j in range(len(inpVars)):
                        if numRuns == 0:
                            sweepDict[inpVars[j]][0] = readData[j]
                        else:
                            sweepDict[inpVars[j]] = np.append(sweepDict[inpVars[j]],readData[j])
                           
                    numRuns += 1

        f.close()
        return sweepDict
    else:
        raise Exception(ctlfile + ' does not exist')

def writeTurbulenceProperties(model,path):
    turbulencePropertiesFile = path + '/turbulenceProperties'
    f = open(turbulencePropertiesFile, 'w')
    turbulenceProperties = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                            '| ========                 |                                                 |\n'
                            '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                            '|  \    /   O peration     | Version:  v1812                                 |\n'
                            '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                            '|    \/     M anipulation  |                                                 |\n'
                            '\*--------------------------------------------------------------------------*/\n'
                            'FoamFile\n'
                            '{\n'
                            '    version     2.0;\n'
                            '    format      ascii;\n'
                            '    class       dictionary;\n'
                            '    location    \"constant\";\n'
                            '    object      turbulenceProperties;\n'
                            '}\n'
                            '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                            '\n'
                            'simulationType RAS;\n'
                            'RAS \n'
                            '{ \n'
                            '    RASModel             '+turbulenceModels[model]+';\n'
                            '    turbulence           on;\n'
                            '    printCoeffs          off;\n'
                            '    nuTildaMin           1e-16;\n'
                            '    Prt                  '+str(PRT)+';\n'
                            '} \n'
                            '\n'
                            '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')

    f.write(turbulenceProperties)
    f.close()

def writeTransportProperties(nu,path):
    transportPropertiesFile = path + '/transportProperties'
    f = open(transportPropertiesFile, 'w')
    transportProperties = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                           '| ========                 |                                                 |\n'
                           '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                           '|  \    /   O peration     | Version:  v1812                                 |\n'
                           '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                           '|    \/     M anipulation  |                                                 |\n'
                           '\*--------------------------------------------------------------------------*/\n'
                           'FoamFile\n'
                           '{\n'
                           '    version     2.0;\n'
                           '    format      ascii;\n'
                           '    class       dictionary;\n'
                           '    location    \"constant\";\n'
                           '    object      transportProperties;\n'
                           '}\n'
                           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                           '\n'
                           'transportModel Newtonian;\n'
                           '\n'
                           'nu '+str(nu)+';\n'
                           'Pr '+str(PR)+';\n'
                           '\n'
                           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')

    f.write(transportProperties)
    f.close()

def writeThermophysicalProperties(mu,T,path):
    thermophysicalPropertiesFile = path + '/thermophysicalProperties'
    f = open(thermophysicalPropertiesFile, 'w')
    thermophysicalProperties = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                                '| ========                 |                                                 |\n'
                                '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                                '|  \    /   O peration     | Version:  v1812                                 |\n'
                                '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                                '|    \/     M anipulation  |                                                 |\n'
                                '\*--------------------------------------------------------------------------*/\n'
                                'FoamFile\n'
                                '{\n'
                                '    version     2.0;\n'
                                '    format      ascii;\n'
                                '    class       dictionary;\n'
                                '    location    \"constant\";\n'
                                '    object      thermophysicalProperties;\n'
                                '}\n'
                                '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                                '\n'
                                'thermoType \n'
                                '{ \n'
                                '    mixture               pureMixture;\n'
                                '    specie                specie;\n'
                                '    equationOfState       perfectGas;\n'
                                '    energy                sensibleInternalEnergy;\n'
                                '    thermo                hConst;\n'
                                '    type                  hePsiThermo;\n'
                                '    transport             const;\n'
                                '} \n'
                                '\n'
                                'mixture \n'
                                '{ \n'
                                '    specie \n'
                                '    { \n'
                                '        molWeight           '+str(MOLW)+'; \n'
                                '    } \n'
                                '    thermodynamics \n'
                                '    { \n'
                                '        Cp                  '+str(RGAS*3.5)+'; \n'
                                '        Hf                  '+str(HFORM)+'; \n'
                                '    } \n'
                                '    transport \n'
                                '    { \n'
                                '        mu                  '+str(mu)+'; \n'
                                '        Pr                  '+str(PR)+'; \n'
                                '        TRef                273.15; \n'
                                '    } \n'
                                '} \n'
                                '\n'
                                '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')

    f.write(thermophysicalProperties)
    f.close()

def writeControlDict(numIter,restart,path):
    startFrom = 'startTime' if restart == 0 else 'latestTime'
    controlDictFile = path + '/controlDict'
    f = open(controlDictFile, 'w')
    controlDict = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                   '| ========                 |                                                 |\n'
                   '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                   '|  \    /   O peration     | Version:  v1812                                 |\n'
                   '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                   '|    \/     M anipulation  |                                                 |\n'
                   '\*--------------------------------------------------------------------------*/\n'
                   'FoamFile\n'
                   '{\n'
                   '    version     2.0;\n'
                   '    format      ascii;\n'
                   '    class       dictionary;\n'
                   '    location    \"system\";\n'
                   '    object      controlDict;\n'
                   '}\n'
                   '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                   '\n'
                   'startFrom       '+startFrom+';\n'
                   'startTime       0;\n'
                   'stopAt          endTime;\n'
                   'endTime         '+str(int(numIter))+';\n'
                   'deltaT          1;\n'
                   'writeControl    timeStep;\n'
                   'writeInterval   '+str(int(numIter))+';\n'
                   'purgeWrite      0;\n'
                   'writeFormat     ascii;\n'
                   'writePrecision  16;\n'
                   'writeCompression on;\n'
                   'timeFormat      general;\n'
                   'timePrecision   16;\n'
                   'runTimeModifiable true;\n'
                   '\n'
                   'DebugSwitches\n'
                   '{\n'
                   '    SolverPerformance 0;\n'
                   '}')

    f.write(controlDict)
    f.close()

def writeDecomposeParDict(path):
    decomposeParDictFile = path + '/decomposeParDict'
    f = open(decomposeParDictFile, 'w')
    decomposeParDict = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                        '| ========                 |                                                 |\n'
                        '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                        '|  \    /   O peration     | Version:  v1812                                 |\n'
                        '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                        '|    \/     M anipulation  |                                                 |\n'
                        '\*--------------------------------------------------------------------------*/\n'
                        'FoamFile\n'
                        '{\n'
                        '    version     2.0;\n'
                        '    format      ascii;\n'
                        '    class       dictionary;\n'
                        '    location    \"system\";\n'
                        '    object      decomposeParDict;\n'
                        '}\n'
                        '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                        '\n'
                        'numberOfSubdomains     4;\n'
                        '\n'
                        'method                 scotch;\n'
                        '\n'
                        'simpleCoeffs \n'
                        '{ \n'
                        '    n                  (2 2 1);\n'
                        '    delta              0.001;\n'
                        '} \n'
                        '\n'
                        'distributed            false;\n'
                        '\n'
                        'roots();\n'
                        '\n'
                        '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')

    f.write(decomposeParDict)
    f.close()

def writeFvSchemes(solver,path):
    fvSchemesFile = path + '/fvSchemes'
    f = open(fvSchemesFile, 'w')
    fvSchemes = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                 '| ========                 |                                                 |\n'
                 '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                 '|  \    /   O peration     | Version:  v1812                                 |\n'
                 '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                 '|    \/     M anipulation  |                                                 |\n'
                 '\*--------------------------------------------------------------------------*/\n'
                 'FoamFile\n'
                 '{\n'
                 '    version     2.0;\n'
                 '    format      ascii;\n'
                 '    class       dictionary;\n'
                 '    location    \"system\";\n'
                 '    object      fvSchemes;\n'
                 '}\n'
                 '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                 '\n'
                 'ddtSchemes \n'
                 '{\n'
                 '    default                                             steadyState;\n'
                 '}\n'
                 '\n'
                 'gradSchemes\n'
                 '{\n'
                 '    default                                             Gauss linear;\n'
                 '}\n'
                 '\n'
                 'interpolationSchemes\n'
                 '{\n'
                 '    default                                             linear;\n'
                 '}\n'
                 '\n'
                 'laplacianSchemes\n'
                 '{\n'
                 '    default                                             Gauss linear corrected;\n'
                 '}\n'
                 '\n'
                 'snGradSchemes\n'
                 '{\n'
                 '    default                                             corrected;\n'
                 '}\n'
                 '\n'
                 'wallDist\n'
                 '{\n'
                 '    method                                              meshWaveFrozen;\n'
                 '}\n'
                 '\n')
                 
    if solver != solvers[2]:
        fvSchemes = fvSchemes + ('divSchemes\n'
                                 '{\n'
                                 '    default                                             none;\n'
                                 '    div(phi,U)                                          bounded Gauss linearUpwindV grad(U);\n'
                                 '    div(phi,e)                                          bounded Gauss upwind;\n'
                                 '    div(phi,h)                                          bounded Gauss upwind;\n'
                                 '    div(phi,T)                                          bounded Gauss upwind;\n'
                                 '    div(phi,nuTilda)                                    bounded Gauss upwind;\n'
                                 '    div(phi,k)                                          bounded Gauss upwind;\n'
                                 '    div(phi,omega)                                      bounded Gauss upwind;\n'
                                 '    div(phi,epsilon)                                    bounded Gauss upwind;\n'
                                 '    div(phi,K)                                          bounded Gauss upwind;\n'
                                 '    div(phi,Ekp)                                        bounded Gauss upwind;\n'
                                 '    div((nuEff*dev2(T(grad(U)))))                       Gauss linear;\n'
                                 '    div(((rho*nuEff)*dev2(T(grad(U)))))                 Gauss linear;\n'
                                 '    div(pc)                                             bounded Gauss upwind;\n'
                                 '}\n'
                                 '\n'
                                 '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')
    else:
        fvSchemes = fvSchemes + ('divSchemes\n'
                                 '{\n'
                                 '    default                                             none;\n'
                                 '    div(phi,U)                                          Gauss linearUpwindV grad(U);\n'
                                 '    div(phi,e)                                          Gauss upwind;\n'
                                 '    div((nuEff*dev2(T(grad(U)))))                       Gauss linear;\n'
                                 '    div(phi,h)                                          Gauss upwind;\n'
                                 '    div(phid,p)                                         Gauss limitedLinear 1.0;\n'
                                 '    div(((rho*nuEff)*dev2(T(grad(U)))))                 Gauss linear;\n'
                                 '    div(phi,nuTilda)                                    Gauss upwind;\n'
                                 '    div(phi,k)                                          Gauss upwind;\n'
                                 '    div(phi,omega)                                      Gauss upwind;\n'
                                 '    div(phi,epsilon)                                    Gauss upwind;\n'
                                 '    div(phi,K)                                          Gauss upwind;\n'
                                 '    div(phi,Ekp)                                        Gauss upwind;\n'
                                 '    div(pc)                                             Gauss upwind;\n'
                                 '}\n'
                                 '\n'
                                 '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')
                
    f.write(fvSchemes)
    f.close()

def writeFvSolution(solver,path):
    fvSolutionFile = path + '/fvSolution'
    f = open(fvSolutionFile,'w')
    fvSolution = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                  '| ========                 |                                                 |\n'
                  '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                  '|  \    /   O peration     | Version:  v1812                                 |\n'
                  '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                  '|    \/     M anipulation  |                                                 |\n'
                  '\*--------------------------------------------------------------------------*/\n'
                  'FoamFile\n'
                  '{\n'
                  '    version     2.0;\n'
                  '    format      ascii;\n'
                  '    class       dictionary;\n'
                  '    object      fvSolution;\n'
                  '}\n'
                  '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                  '\n'
                  'SIMPLE\n'
                  '{\n'
                  '    nNonOrthogonalCorrectors           0;\n'
                  '}\n'
                  '\n'
                  'solvers\n'
                  '{\n'
                  '    \"(p|p_rgh|G)\"\n'
                  '    {\n'
                  '        \n'
                  '        solver                         GAMG;\n'
                  '        smoother                       GaussSeidel;\n'
                  '        relTol                         0.1;\n'
                  '        tolerance                      0;\n'
                  '    }\n'
                  '    Phi\n'
                  '    {\n'
                  '        $p;\n'
                  '        relTol                         0;\n'
                  '        tolerance                      1e-6;\n'
                  '    }\n'
                  '    \"(U|T|e|h|nuTilda|k|omega|epsilon)\"\n'
                  '    {\n'
                  '        solver                         smoothSolver;\n'
                  '        smoother                       GaussSeidel;\n'
                  '        relTol                         0.1;\n'
                  '        tolerance                      0;\n'
                  '        nSweeps                        1;\n'
                  '    }\n'
                  '}\n'
                  'potentialFlow\n'
                  '{\n'
                  '    nNonOrthogonalCorrectors           20;\n'
                  '}\n')

    if solver!=solvers[2]:
        fvSolution = fvSolution + ('relaxationFactors\n'
                                   '{\n'
                                   '    fields\n'
                                   '    {\n'
                                   '        \"(p|p_rgh)\"                         0.30;\n'
                                   '        rho                                 0.10;\n'
                                   '    }\n'
                                   '    equations\n'
                                   '    {\n'
                                   '        \"(U|T|e|h|nuTilda|k|epsilon|omega)\" 0.70;\n'
                                   '    }\n'
                                   '\n'
                                   '}\n')
    else:
        fvSolution = fvSolution + ('relaxationFactors\n'
                                   '{\n'
                                   '    fields\n'
                                   '    {\n'
                                   '        \"(p|p_rgh|rho)\"                     1.0;\n'
                                   '    }\n'
                                   '    equations\n'
                                   '    {\n'
                                   '        p                                     1.0;\n'
                                   '        \"(U|T|e|h|nuTilda|k|epsilon|omega)\" 0.80;\n'
                                   '    }\n'
                                   '\n'
                                   '}\n')

    f.write(fvSolution)
    f.close()
    
def writeChangeDictionary(changeDict,bcs,path):
    changes = {}
    for key in changeDict.keys():
        changes[key] = {}
        if key in ['U']:
            value = 'uniform ('+' '.join([str(i) for i in changeDict[key]])+')'
        else:
            value = 'uniform ' + str(changeDict[key])
            
        if key in ['nut', 'alphat']:
            changes[key]['wall'] = {}
            changes[key]['wall']['value'] = value
            
        if key in ['U','T','p','nut','nuTilda','alphat']:
            changes[key]['patch'] = {}
            if key in ['U','T','nuTilda']: changes[key]['patch']['inletValue'] = value
            changes[key]['patch']['value'] = value
            
    
    changeDictionaryFile = path + '/changeDictionaryDict'
    f = open(changeDictionaryFile,'w')
    changeDictionary = ('/*--------------------------------*- C++ -*---------------------------------*\\\n'
                        '| ========                 |                                                 |\n'
                        '| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
                        '|  \    /   O peration     | Version:  v1812                                 |\n'
                        '|   \  /    A nd           | Web:      www.OpenFOAM.com                      |\n'
                        '|    \/     M anipulation  |                                                 |\n'
                        '\*--------------------------------------------------------------------------*/\n'
                        'FoamFile\n'
                        '{\n'
                        '    version     2.0;\n'
                        '    format      ascii;\n'
                        '    class       dictionary;\n'
                        '    location    \"system\";\n'
                        '    object      changeDictionaryDict;\n'
                        '}\n\n')

    for var in changes.keys():
        changeDictionary = changeDictionary + (var+'\n'
                                               '{\n'
                                               '    boundaryField\n'
                                               '    {\n')
        for boundary in changes[var].keys():
            changeDictionary = changeDictionary + ('        '+'|'.join([i for i in bcs[bcs[:,1]==boundary,0]])+'\n'
                                                   '        {\n')
            for change in changes[var][boundary].keys():
                changeDictionary = changeDictionary + ('            '+change+' '+changes[var][boundary][change]+';\n')
            changeDictionary = changeDictionary + ('        }\n')
        changeDictionary = changeDictionary + ('    }\n'
                                               '}\n\n')
                                           
    f.write(changeDictionary)
    f.close()
    

def writeBCU(U,bcs,path):
    bcUFile = path + '/U'
    f = open(bcUFile,'w')
    bcU = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  v1812                                 |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volVectorField;\n'
           '    location    \"0\";\n'
           '    object      U;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [0 1 -1 0 0 0 0];\n'
           '\n'
           'internalField uniform ('+' '.join([str(i) for i in U])+');\n'
           '\n'
           'boundaryField\n'
           '{\n')
   
    for i in range(len(bcs[:,0])):
        bcU = bcU + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcU = bcU + ('        type            inletOutlet;\n'
                         '        inletValue      $internalField;\n'
                         '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcU = bcU + ('        type            fixedValue;\n'
                         '        value           uniform (0 0 0);\n')
        elif bcs[i,1]=='symmetry':
            bcU = bcU + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryPlane':
            bcU = bcU + '        type            symmetryPlane;\n'
           
        bcU = bcU + '    }\n'
       
    bcU = bcU + ('}\n'
                 '\n'
                 '// ************************************************************************* //')

    f.write(bcU)
    f.close()

def writeBCT(T,bcs,path):
    bcTFile = path + '/T'
    f = open(bcTFile,'w')
    bcT = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      T;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //'
           '\n'
           'dimensions      [0 0 0 1 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(T)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcT = bcT + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcT = bcT + ('        type            inletOutlet;\n'
                         '        inletValue      $internalField;\n'
                         '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcT = bcT + '        type            zeroGradient;\n'
        elif bcs[i,1]=='symmetry':
            bcT = bcT + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryPlane':
            bcT = bcT + '        type            symmetryPlane;\n'

        bcT = bcT + '    }\n'

    bcT = bcT + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcT)
    f.close()

def writeBCP(P,bcs,path):
    dim = '[0 2 -2 0 0 0 0]' if P==0 else '[1 -1 -2 0 0 0 0]'
    
    bcPFile = path + '/p'
    f = open(bcPFile,'w')
    bcP = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      p;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      '+dim+';\n'
           '\n'
           'internalField   uniform '+str(P)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcP = bcP + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcP = bcP + ('        type            fixedValue;\n'
                         '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcP = bcP + '        type            zeroGradient;\n'
        elif bcs[i,1]=='symmetry':
            bcP = bcP + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryPlane':
            bcP = bcP + '        type            symmetryPlane;\n'
       
        bcP = bcP + '    }\n'
           
    bcP = bcP + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcP)
    f.close()

def writeBCNuTilda(nuTilda,bcs,path):
    bcNuTildaFile = path + '/nuTilda'
    f = open(bcNuTildaFile,'w')
    bcNuTilda = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      nuTilda;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [0 2 -1 0 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(nuTilda)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcNuTilda = bcNuTilda + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcNuTilda = bcNuTilda + ('        type            inletOutlet;\n'
                                     '        inletValue      $internalField;\n'
                                     '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcNuTilda = bcNuTilda + ('        type            fixedValue;\n'
                                     '        value           uniform 0.0;\n')
        elif bcs[i,1]=='symmetry':
            bcNuTilda = bcNuTilda + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetrynuTildalane':
            bcNuTilda = bcNuTilda + '        type            symmetryPlane;\n'
       
        bcNuTilda = bcNuTilda + '    }\n'
           
    bcNuTilda = bcNuTilda + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcNuTilda)
    f.close()
   
def writeBCNut(nut,bcs,path):
    bcNutFile = path + '/nut'
    f = open(bcNutFile,'w')
    bcNut = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      nut;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [0 2 -1 0 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(nut)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcNut = bcNut + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcNut = bcNut + ('        type            calculated;\n'
                             '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcNut = bcNut + ('        type            nutUSpaldingWallFunction;\n'
                             '        value           $internalField;\n')
        elif bcs[i,1]=='symmetry':
            bcNut = bcNut + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryNutlane':
            bcNut = bcNut + '        type            symmetryPlane;\n'
       
        bcNut = bcNut + '    }\n'
           
    bcNut = bcNut + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcNut)
    f.close()
   
def writeBCAlphat(alphat,bcs,path):
    bcAlphatFile = path + '/alphat'
    f = open(bcAlphatFile,'w')
    bcAlphat = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      alphat;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [1 -1 -1 0 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(alphat)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcAlphat = bcAlphat + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcAlphat = bcAlphat + ('        type            calculated;\n'
                                   '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcAlphat = bcAlphat + ('        type            compressible::alphatWallFunction;\n'
                                   '        value           $internalField;\n')
        elif bcs[i,1]=='symmetry':
            bcAlphat = bcAlphat + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryAlphatlane':
            bcAlphat = bcAlphat + '        type            symmetryPlane;\n'
       
        bcAlphat = bcAlphat + '    }\n'
           
    bcAlphat = bcAlphat + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcAlphat)
    f.close()
   
def writeBCK(k,bcs,path):
    bcKFile = path + '/k'
    f = open(bcKFile,'w')
    bcK = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      k;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [0 2 -2 0 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(k)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcK = bcK + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcK = bcK + ('        type            inletOutlet;\n'
                         '        inletValue      $internalField;\n'
                         '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcK = bcK + ('        type            kqRWallFunction;\n'
                         '        value           $internalField;\n')
        elif bcs[i,1]=='symmetry':
            bcK = bcK + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryKlane':
            bcK = bcK + '        type            symmetryPlane;\n'
       
        bcK = bcK + '    }\n'
           
    bcK = bcK + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcK)
    f.close()
   
def writeBCOmega(omega,bcs,path):
    bcOmegaFile = path + '/omega'
    f = open(bcOmegaFile,'w')
    bcOmega = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      omega;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [0 0 -1 0 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(omega)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcOmega = bcOmega + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcOmega = bcOmega + ('        type            inletOutlet;\n'
                                 '        inletValue      $internalField;\n'
                                 '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcOmega = bcOmega + ('        type            omegaWallFunction;\n'
                                 '        value           $internalField;\n'
                                 '        blended         true;\n')
        elif bcs[i,1]=='symmetry':
            bcOmega = bcOmega + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryOmegalane':
            bcOmega = bcOmega + '        type            symmetryPlane;\n'
       
        bcOmega = bcOmega + '    }\n'
           
    bcOmega = bcOmega + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcOmega)
    f.close()

def writeBCEpsilon(epsilon,bcs,path):
    bcEpsilonFile = path + '/epsilon'
    f = open(bcEpsilonFile,'w')
    bcEpsilon = ('/*--------------------------------*- C++ -*----------------------------------*\\\n'
           '| =========                 |                                                 |\n'
           '| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n'
           '|  \\    /   O peration     | Version:  plus                                  |\n'
           '|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n'
           '|    \\/     M anipulation  |                                                 |\n'
           '\*---------------------------------------------------------------------------*/\n'
           'FoamFile\n'
           '{\n'
           '    version     2.0;\n'
           '    format      ascii;\n'
           '    class       volScalarField;\n'
           '    location    \"0\";\n'
           '    object      epsilon;\n'
           '}\n'
           '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
           '\n'
           'dimensions      [0 2 -3 0 0 0 0];\n'
           '\n'
           'internalField   uniform '+str(epsilon)+';\n'
           '\n'
           'boundaryField\n'
           '{\n')

    for i in range(len(bcs[:,0])):
        bcEpsilon = bcEpsilon + ('    '+bcs[i,0]+'\n'
                     '    {\n')
        if bcs[i,1]=='patch':
            bcEpsilon = bcEpsilon + ('        type            inletOutlet;\n'
                                     '        inletValue      $internalField;\n'
                                     '        value           $internalField;\n')
        elif bcs[i,1]=='wall':
            bcEpsilon = bcEpsilon + ('        type            epsilonWallFunction;\n'
                                     '        value           $internalField;\n')
        elif bcs[i,1]=='symmetry':
            bcEpsilon = bcEpsilon + '        type            symmetry;\n'
        elif bcs[i,1]=='symmetryEpsilonlane':
            bcEpsilon = bcEpsilon + '        type            symmetryPlane;\n'
       
        bcEpsilon = bcEpsilon + '    }\n'
           
    bcEpsilon = bcEpsilon + ('}\n'
                 '\n'
                 '// ************************************************************************* //')
    f.write(bcEpsilon)
    f.close()
   
def writeRunScript(runScriptFile,solver,method,bcs,A0,alpha,U,mach,rho,Re,targetCL):
    patches = [i for i in bcs[bcs[:,1]=='patch',0]]
    walls = [i for i in bcs[bcs[:,1]=='wall',0]]
    solveCL = (method==2)
    designSurfaces = str(walls) if solveCL else ''
    SF = str(1.0/(0.5*rho*U*U*A0))
    alphaStr = str(alpha) if not solveCL else '0'

    f = open(runScriptFile,'w')
    runScript = ('import os\n'
                 'from mpi4py import MPI\n'
                 'from dafoam import PYDAFOAM, optFuncs\n'
                 'from pygeo import *\n'
                 'from idwarp import *\n'
                 'import numpy as np\n\n')
         
    #Note: alpha currently set to XZ plane
    runScript = runScript + ('def alpha(val):\n'
                             '    aoa = val * np.pi / 180.0\n'
                             '    inletU = [float('+str(U)+' * np.cos(aoa)), 0.0, float('+str(U)+' * np.sin(aoa))]\n'
                             '    dragDir = [float(np.cos(aoa)), 0.0, float(np.sin(aoa))]\n'
                             '    liftDir = [float(-np.sin(aoa)), 0.0, float(np.cos(aoa))]\n'
                             '    return inletU,dragDir,liftDir\n\n')
                
    if solveCL:
        runScript = runScript + ('def constCL(CLStar, DASolver, alpha0=0, dCl=0.2, tol=1e-4, maxit=20):\n'
                                 '    aoa = alpha0\n'
                                 '    CL = 0\n'
                                 '    output = {}\n'
                                 '    evalFuncs = [\'CD\',\'CL\',\'CX\',\'CY\',\'CZ\',\'CMX\',\'CMY\',\'CMZ\']\n'
                                 '    for i in range(maxit):\n'
                                 '        inletU,dragDir,liftDir = alpha(aoa)\n'
                                 '        DASolver.setOption(\'primalBC\', {\'U0\': {\'variable\': \'U\', \'patches\': '+str(patches)+', \'value\': inletU}})\n'
                                 '        DASolver.updateDAOption()\n'
                                 '        DASolver()\n'
                                 '        DASolver.evalFunctions(output, evalFuncs=evalFuncs)\n'
                                 '        CX = output[\'CX\']\n'
                                 '        CZ = output[\'CZ\']\n'
                                 '        CL = -CX*np.sin(aoa*np.pi/180.0)+CZ*np.cos(aoa*np.pi/180.0)\n'
                                 '        CD = CX*np.cos(aoa*np.pi/180.0)+CZ*np.sin(aoa*np.pi/180.0)\n'
                                 '        if abs(CLStar-CL)/CLStar<tol:\n'
                                 '            return aoa,CL,CD,output\n'
                                 '        aoa += (CLStar-CL)/dCl\n'
                                 '        print(\'\\nAngleOfAttack : \' + str(aoa) + \'\\n\')\n'
                                 '    return aoa,CL,CD,output\n\n')

    runScript = runScript + ('inletU,dragDir,liftDir=alpha('+alphaStr+')\n\n'
                             ''
                             'daOptions = {\n'
                             '    \'designSurfaces\': '+designSurfaces+',\n'
                             '    \'solverName\': \''+solver+'\',\n'
                             '    \'useAD\': {\'mode\': \'reverse\'},\n'
                             '    \'primalMinResTol\': '+str(PRIMAL_MIN_RES_TOL)+',\n'
                             '    \'primalBC\': {\n'
                             '        \'useWallFunction\': '+WALL_FUNCTION+',\n'
                             '    },\n'
                             '    \'primalVarBounds\': {\n'
                             '        \'UMax\': 1000.0,\n'
                             '        \'UMin\': -1000.0,\n'
                             '        \'pMax\': 500000.0,\n'
                             '        \'pMin\': 20000.0,\n'
                             '        \'eMax\': 500000.0,\n'
                             '        \'eMin\': 100000.0,\n'
                             '        \'rhoMax\': 5.0,\n'
                             '        \'rhoMin\': 0.2,\n'
                             '    },\n'
                             '    \'objFunc\': {\n'
                             '        \'CD\': {\n'
                             '            \'part1\': {\n'
                             '                \'type\': \'force\',\n'
                             '                \'source\': \'patchToFace\',\n'
                             '                \'patches\': '+str(walls)+',\n'
                             '                \'directionMode\': \'fixedDirection\',\n'
                             '                \'direction\': dragDir,\n'
                             '                \'scale\': '+SF+',\n'
                             '                \'addToAdjoint\': True,\n'
                             '            }\n'
                             '        },\n'
                             '        \'CL\': {\n'
                             '            \'part1\': {\n'
                             '                \'type\': \'force\',\n'
                             '                \'source\': \'patchToFace\',\n'
                             '                \'patches\': '+str(walls)+',\n'
                             '                \'directionMode\': \'fixedDirection\',\n'
                             '                \'direction\': liftDir,\n'
                             '                \'scale\': '+SF+',\n'
                             '                \'addToAdjoint\': True,\n'
                             '            }\n'
                             '        },\n'
                             '        \'CX\': {\n'
                             '            \'part1\': {\n'
                             '                \'type\': \'force\',\n'
                             '                \'source\': \'patchToFace\',\n'
                             '                \'patches\': '+str(walls)+',\n'
                             '                \'directionMode\': \'fixedDirection\',\n'
                             '                \'direction\': [1.0, 0.0, 0.0],\n'
                             '                \'scale\': '+SF+',\n'
                             '                \'addToAdjoint\': True,\n'
                             '            }\n'
                             '        },\n'
                             '        \'CY\': {\n'
                             '            \'part1\': {\n'
                             '                \'type\': \'force\',\n'
                             '                \'source\': \'patchToFace\',\n'
                             '                \'patches\': '+str(walls)+',\n'
                             '                \'directionMode\': \'fixedDirection\',\n'
                             '                \'direction\': [0.0, 1.0, 0.0],\n'
                             '                \'scale\': '+SF+',\n'
                             '                \'addToAdjoint\': True,\n'
                             '            }\n'
                             '        },\n'
                             '        \'CZ\': {\n'
                             '            \'part1\': {\n'
                             '                \'type\': \'force\',\n'
                             '                \'source\': \'patchToFace\',\n'
                             '                \'patches\': '+str(walls)+',\n'
                             '                \'directionMode\': \'fixedDirection\',\n'
                             '                \'direction\': [0.0, 0.0, 1.0],\n'
                             '                \'scale\': '+SF+',\n'
                             '                \'addToAdjoint\': True,\n'
                             '            }\n'
                             '        },\n'
                             '        \'CMX\': {\n'
                             '            \'part1\': {\n'
                             '                 \'type\': \'moment\',\n'
                             '                 \'source\': \'patchToFace\',\n'
                             '                 \'patches\': '+str(walls)+',\n'
                             '                 \'axis\': [1.0, 0.0, 0.0],\n'
                             '                 \'center\': [0.25, 0.0, 0.0],\n'
                             '                 \'scale\': '+SF*D+',\n'
                             '                 \'addToAdjoint\': True,\n'
                             '             }\n'
                             '         },\n'
                             '         \'CMY\': {\n'
                             '            \'part1\': {\n'
                             '                 \'type\': \'moment\',\n'
                             '                 \'source\': \'patchToFace\',\n'
                             '                 \'patches\': '+str(walls)+',\n'
                             '                 \'axis\': [0.0, 1.0, 0.0],\n'
                             '                 \'center\': [0.25, 0.0, 0.0],\n'
                             '                 \'scale\': '+SF*D+',\n'
                             '                 \'addToAdjoint\': True,\n'
                             '             }\n'
                             '         },\n'
                             '         \'CMZ\': {\n'
                             '            \'part1\': {\n'
                             '                 \'type\': \'moment\',\n'
                             '                 \'source\': \'patchToFace\',\n'
                             '                 \'patches\': '+str(walls)+',\n'
                             '                 \'axis\': [0.0, 0.0, 1.0],\n'
                             '                 \'center\': [0.25, 0.0, 0.0],\n'
                             '                 \'scale\': '+SF*D+',\n'
                             '                 \'addToAdjoint\': True,\n'
                             '             }\n'
                             '         },\n'
                             '    },\n'
                             '    \'checkMeshThreshold\': {\'maxAspectRatio\': '+str(MAX_AR)+'},\n'
                             '    \'designVar\': {},\n'
                             '}\n'
                             '\n'
                             'DASolver = PYDAFOAM(options=daOptions, comm=MPI.COMM_WORLD)\n\n')
                             
    if solveCL:
        runScript = runScript + ('alpha,CL,CD,output=constCL('+str(targetCL)+',DASolver)\n'
                                 'if MPI.COMM_WORLD.rank == 0:\n'
                                 '    if os.path.exists(\'../foam.output\'):\n'
                                 '        f = open(\'../foam.output\',\'a\')\n'
                                 '    else:\n'
                                 '        f = open(\'../foam.output\',\'w\')\n'
                                 '        f.write(\'AOA MACH REYNOLDS CL CD CMX CMY CMZ\\n\')\n'
                                 '    f.write(str(alpha)+\' \'+str('+str(mach)+')+\' \'+str('+str(Re)+')+\' \'+str(CL)+\' \'+str(CD)+\' \'+str(output[\'CMX\'])+\' \'+str(output[\'CMY\'])+\' \'+str(output[\'CMZ\'])+\'\\n\')\n'
                                 '    f.close()\n')
    else:
        runScript = runScript + ('DASolver()\n'
                                 'funcs = {}\n'
                                 'evalFuncs = [\'CD\',\'CL\',\'CX\',\'CY\',\'CZ\',\'CMX\',\'CMY\',\'CMZ\']\n'
                                 'DASolver.evalFunctions(output, evalFuncs)\n'
                                 'if MPI.COMM_WORLD.rank == 0:\n'
                                 '    if os.path.exists(\'../foam.output\'):\n'
                                 '        f = open(\'../foam.output\',\'a\')\n'
                                 '    else:\n'
                                 '        f = open(\'../foam.output\',\'w\')\n'
                                 '        f.write(\'AOA MACH REYNOLDS CL CD CMX CMY CMZ\\n\')\n'
                                 '    f.write(str(alpha)+\' \'+str('+str(mach)+')+\' \'+str('+str(Re)+')+\' \'+str(output[\'CL\'])+\' \'+str(output[\'CD\'])+\' \'+str(output[\'CMX\'])+\' \'+str(output[\'CMY\'])+\' \'+str(output[\'CMZ\'])+\'\\n\')\n'
                                 '    f.close()\n')

    f.write(runScript)
    f.close()
    
def writeJobScript(caseDict,newDirPath,newDirName,prevJob,oldIter,dependID):
    cores = str(int(args.nodes*ppn))
    jobFile = newDirPath+'/'+newDirName+'.job'
    f = open(jobFile,'w')
    
    jobScript = ('#!/bin/bash\n\n'
                 '#PBS -N '+newDirName+'\n'
                 '#PBS -q '+args.queue+'\n'
                 '#PBS -d '+newDirPath+'\n'
                 '#PBS -o '+newDirName+'.o'+'\n'
                 '#PBS -e '+newDirName+'.e'+'\n'
                 '#PBS -l nodes='+str(int(args.nodes))+':ppn='+str(int(ppn))+'\n\n')
                 
    if dependID != None:
        jobScript = jobScript + ('#PBS -W depend=afterany:' + str(dependID) + '\n')
            
    jobScript = jobScript + ('source ~athomas/bin/loadDAFoam.sh\n\n')
    
    if 'RESTART' in caseDict.keys() and caseDict['RESTART'][i]==2:
        jobScript = jobScript + ('latestTime=$(cd ../'+prevJob+' && foamListTimes -latestTime | tail -n 1)\n'
                                 'cp -r ../' + prevJob + '/' + '$latestTime ' + '0\n'
                                 'rm 0/uniform/time\n'
                                 'changeDictionary -latestTime\n\n')           
    if 'RESTART' in caseDict.keys() and caseDict['RESTART'][i]==1:
        jobScript = jobScript + ('rm -r 0 && cp -r ' + str(int(oldIter)) + ' ' + '0\n'
                                 'rm 0/uniform/time\n'
                                 'rm -r ' + str(int(oldIter)) + '\n\n')  

    jobScript = jobScript + ('mpirun -np '+cores+' python '+runScrName+' > '+newDirName+'.txt\n'
                             'reconstructPar\n'
                             'rm -r processor*')
    f.write(jobScript)
    f.close()
    


### Create Sweep
sweepDict = readCtl()
if 'TEMP' in sweepDict.keys():
    if 'MACH' in sweepDict.keys():
        if 'RE' in sweepDict.keys():
            if 'ALPHA' in sweepDict.keys():
                method = 1
            elif 'CL' in sweepDict.keys():
                method = 2
            else:
                raise Exception('Invalid Input')
        else:
            raise Exception('Invalid Input')
    else:
        raise Exception('Invalid Input')
else:
    raise Exception('Invalid Input')

first = args.start or 0
last = args.end or numRuns
dirNames = np.array([args.baseName + '_' + '_'.join([inp + '_' + str(sweepDict[inp][i]) for inp in inpVars if (inp not in ['RESTART','NUMITER'])]) for i in range(numRuns)[first:last]])
prevID = None
bcs = getBC()
cases = []

try:
    subprocess.check_call('which qsub >/dev/null 2>&1', shell=True) # raises exception if qsub not found
except:
    queue = None
else:
    queue = None if args.queue == 'none' else args.queue

for i in range(numRuns)[first:last]:
    #Calculate conditions from input
    if method==1:
        targetCL = None
        alpha = sweepDict['ALPHA'][i]
        T = sweepDict['TEMP'][i]
        mu = sutherland(T)
        Ma = sweepDict['MACH'][i]
        U = mach(Ma,T)
        Re = sweepDict['RE'][i]
        rho = reynoldsNum(Re,mu,U,D)
        nu = mu/rho
        P = perfectGas(rho,T,'P') if Ma>=0.1 else 0
        nu_t, alpha_t, nuTilda, k, omega, epsilon = turb(nu,U)
    elif method==2:
        targetCL = sweepDict['CL'][i]
        alpha = 0
        T = sweepDict['TEMP'][i]
        mu = sutherland(T)
        Ma = sweepDict['MACH'][i]
        U = mach(Ma,T)
        Re = sweepDict['RE'][i]
        rho = reynoldsNum(Re,mu,U,D)
        nu = mu/rho
        P = perfectGas(rho,T,'P') if Ma>=0.1 else 0
        nu_t, alpha_t, nuTilda, k, omega, epsilon = turb(nu,U)
        
    else:
        raise Exception('Error?')
    
    if ('RESTART' in sweepDict.keys() and sweepDict['RESTART'][i]==3):
        if len(cases) > 0:
            cases.append(next(case for case in cases if case[0]==dirNames[i]))
        else:
            raise Exception('Cannot Restart From Nothing')
    else:
        #Get Other Parameters
        solver = chooseSolver(Ma)
    
        #Create Case Directory
        newDirName = dirNames[i]
        newDirPath = args.rootdir + '/' + newDirName
        if os.path.exists(newDirPath): delDir(newDirPath)
        os.system('mkdir ' + newDirPath)

        ##Copy Data into new Case File, create if not already
        #Constant Dir
        constantDir = args.rootdir + '/constant'
        if os.path.exists(constantDir):
            copyDir(constantDir, newDirPath + '/constant')
        else:
            os.system('mkdir ' + newDirPath + '/constant')
            
        if solver==solvers[0]:
            writeTransportProperties(nu,newDirPath + '/constant')
        else:
            writeThermophysicalProperties(mu,T,newDirPath + '/constant')
            
        writeTurbulenceProperties(args.turbModel, newDirPath + '/constant')
    
        #System Dir
        systemDir = args.rootdir + '/system'
        if os.path.exists(systemDir):
            copyDir(systemDir, newDirPath + '/system')
        else:
            os.system('mkdir ' + newDirPath + '/system')
        
        nIter = sweepDict['NUMITER'][i]
        writeDecomposeParDict(newDirPath + '/system')
        writeFvSchemes(solver,newDirPath + '/system')
        writeFvSolution(solver,newDirPath + '/system')

        #0 Dir
        os.system('mkdir ' + newDirPath + '/0.orig')
      
        if Ma>=0.1:
            writeBCAlphat(alpha_t, bcs, newDirPath + '/0.orig')
            writeBCEpsilon(epsilon, bcs, newDirPath + '/0.orig')
        writeBCK(k, bcs, newDirPath + '/0.orig')
        writeBCNuTilda(nuTilda, bcs, newDirPath + '/0.orig')
        writeBCNut(nu_t, bcs, newDirPath + '/0.orig')
        writeBCOmega(omega, bcs, newDirPath + '/0.orig')
        writeBCP(P, bcs, newDirPath + '/0.orig')
        writeBCT(T, bcs, newDirPath + '/0.orig')
        writeBCU(getUVec(U,alpha), bcs, newDirPath + '/0.orig')
    
        #Restarts
        # 0=Normal, 1=Continue Run, 2=Run From Previous, 3=Load Data
        if sweepDict['RESTART'][i] == 0 or 'RESTART' not in sweepDict.keys():
            copyDir(newDirPath + '/0.orig', newDirPath + '/0')
        elif sweepDict['RESTART'][i] == 2:
            changeDict = {'U' : getUVec(U,alpha),
                          'T' : T,
                          'p' : P}
            
            if args.turbModel in ['SA', 'SAFV3']:
                changeDict['nuTilda'] = nuTilda
                changeDict['nut'] = nu_t
                changeDict['alphat'] = alpha_t
            #'omega' : omega,
            #'k' : k,
            #'epsilon' : epsilon
            
            writeChangeDictionary(changeDict, bcs, newDirPath+'/system')
            
        writeControlDict(nIter,sweepDict['RESTART'][i],newDirPath + '/system')
                
        #Other Files
        pvDir = args.rootdir + '/paraview.foam'
        if os.path.exists(pvDir):
            copyFile(pvDir, newDirPath + '/paraview.foam')
        else:
            f = open(newDirPath + '/paraview.foam', 'w')
            f.close()

        writeRunScript(newDirPath+'/'+runScrName,solver,method,bcs,A0,alpha,U,Ma,rho,Re,targetCL)

        #Check For Case Errors

        #Create jobs
        prevJobName = cases[-1][0] if i>0 else None
        oldIter = cases[-1][1] if i>0 else None
        dependID = cases[-1][2] if (i>0 and sweepDict['RESTART'][i] in [1,2]) else None
        writeJobScript(sweepDict,newDirPath,newDirName,prevJobName,oldIter,dependID)
        subprocess.call(['chmod', 'ug+x', newDirPath+'/'+newDirName+'.job'])
        
        #Submit Jobs
        if args.nodes>0:
            qsub = ['qsub', newDirPath+'/'+newDirName+'.job']
            jobid = subprocess.check_output(qsub).strip('.les')
            print('submitted ' + str(jobid))
        else:
            jobid = 0
        cases.append([dirNames[i],nIter,jobid]) #Case Name, nIter, jobid