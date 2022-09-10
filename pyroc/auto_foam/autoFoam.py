import os
import numpy as np
from collections import OrderedDict

class FOAMOPTION(object):
    #Class to store options and their defaults for OpenFOAM

    def __init__(self):
        #Initialize default options:
        self.universalConstants = {
            'RUniversal': 8.3144626, #Universal Gas Constant (J/mol*K)
            'Prt': 0.85, #Turbulent Prandtl Number
            'Pr': 0.72, #Prandtl Number
            'TRef': 273.15, #Reference Temperature, K
        }

        self.speciesDict = {
            'air': {
                'molWeight': 28.964, #g/mol
                'molDOF': 5, #Molecular degrees of freedom (diatomic)
                'Hf': 0.0, #Heat of formation?
            },
        }

        self.dirDict = {
            'rootDir': os.getcwd(), #Directory to write files in
            'sysDir': 'system', #System directory
            'constDir': 'constant', #Constant directory
            'meshDir': 'polyMesh', #Constant/polyMesh directory
            'bcDir': '0', #Initial time directory
        }

        self.refDict = {
            'refChord': 1.0,
            'refArea': 1.0,
            'refLocation': [0.25, 0.0, 0.0],
        }

        self.controlDict = {
            'restart': 0,
            'deltaT': 1,
            'numIter': 1000,
            'printInterval': 1000,
            'precision': 16,
        }

        self.fvSchemes = {}

        self.fvSolution = {}

        ## decomposeParDict option. This file will be automatically written such that users
        ## can run optimization with any number of CPU cores without the need to manually
        ## change decomposeParDict
        self.decomposeParDict = {
            'method': 'scotch',
            'simpleCoeffs': {
                'n': [2, 2, 1], 
                'delta': 0.001,
                },
            'preservePatches': ['None'],
            'singleProcessorFaceSets': ['None'],
        }

        self.blockMeshDict = {}

        self.surfaceFeatureExtractDict = {}

        self.snappyHexMeshDict = {}

        #TODO Implement Janaf, mixing, sutherlandTransport, etc?
        self.thermodynamicsDict = {
            'mixture': 'pureMixture',
            'specie': 'air',
            'equationOfState': 'perfectGas',
            'energy': 'sensibleInternalEnergy',
            'thermo': 'hConst',
            'type': 'hePsiThermo',
            'transport': 'const',
            'transportModel': 'Newtonian',
        }

        self.turbulenceDict = {
            'model': 'SpalartAllmaras',
            'wallFunctions': False,
            'TVR': 3.0, #Turbulent Viscosity Ratio
            'TIR': 0.01, #Turbulent Intensity Ratio
        }

        self.boundaryDict = {
            'U': {'type': 'volVectorField',
                'dimensions': [0, 1, -1, 0, 0, 0, 0],
                'wall': 'fixedZero',
                'patch': 'inletOutlet',
                'value': np.array([[0, 0, 0]])},
            'p': {'type': 'volScalarField',
                'dimensions': [1, -1, -2, 0, 0, 0, 0],
                'wall': 'zeroGradient',
                'patch': 'fixedValue',
                'value': np.array([101325])},
            'T': {'type': 'volScalarField',
                'dimensions': [0, 0, 0, 1, 0, 0, 0],
                'wall': 'zeroGradient',
                'patch': 'inletOutlet',
                'value': np.array([273.15])},
            'nut': {'type': 'volScalarField',
                'dimensions': [0, 2, -1, 0, 0, 0, 0],
                'wall': 'nutUSpaldingWallFunction',
                'patch': 'calculated',
                'value': np.array([4.5e-5])},
            'alphat': {'type': 'volScalarField',
                'dimensions': [1, -1, -1, 0, 0, 0, 0],
                'wall': 'alphatWallFunction',
                'patch': 'calculated',
                'value': np.array([1e-4])},
            'nuTilda': {'type': 'volScalarField',
                'dimensions': [0, 2, -1, 0, 0, 0, 0],
                'wall': 'fixedZero',
                'patch': 'inletOutlet',
                'value': np.array([4.5e-5])},
            'k': {'type': 'volScalarField',
                'dimensions': [0, 2, -2, 0, 0, 0, 0],
                'wall': 'kqRWallFunction',
                'patch': 'inletOutlet',
                'value': np.array([1.5])},
            'epsilon': {'type': 'volScalarField',
                'dimensions': [0, 2, -3, 0, 0, 0, 0],
                'wall': 'epsilonWallFunction',
                'patch': 'inletOutlet',
                'value': np.array([1350])},
            'omega': {'type': 'volScalarField',
                'dimensions': [0, 0, -1, 0, 0, 0, 0],
                'wall': 'omegaWallFunction',
                'patch': 'calculated',
                'value': np.array([1e4])},
        }

        self.solverDict = {
            'DASimpleFoam': ['U', 'p', 'nut', 'nuTilda', 'k', 'epsilon', 'omega'],
            'DARhoSimpleFoam': ['U', 'p', 'T', 'nut', 'alphat', 'nuTilda', 'k', 'epsilon', 'omega'],
            'DARhoSimpleCFoam': ['U', 'p', 'T', 'nut', 'alphat', 'nuTilda', 'k', 'epsilon', 'omega'],
        }

class AutoFOAM(object):
    #Class to write openfoam files

    def __init__(self, options=None):

        self.name = "AUTOFOAM"

        self._initializeOptions(options)

        self.solver = None

        self.boundaries = {}

        self.stateDict = {}

    def updateStateDict(self, mode=0, eqnMode=0):
        if mode == 0: #Default mode is Mach, Reynolds, Temp
            if eqnMode == 0: #Default equation mode is [constGamma, sutherland, perfectGas]
                self.stateDict['U'] = self._mach(self.stateDict['MACH'], self.stateDict['T'])
                self.stateDict['NU'] = self._reynolds(self.stateDict['REYNOLDS'], self.stateDict['U'], self.getOption('refDict')['refChord'])
                self.stateDict['MU'] = self._sutherland(self.stateDict['T'])
                self.stateDict['RHO'] = self.stateDict['MU']/self.stateDict['NU']
                self.stateDict['P'] = self._perfectGas(self.stateDict['RHO'], self.stateDict['T'])
            elif eqnMode == 1: #Coolprop/janaf
                pass
            #Turbulence parameters
            self.stateDict['NUT'] = self.stateDict['NU']*self.getOption('turbulenceDict')['TVR']
            self.stateDict['ALPHAT'] = self.stateDict['NUT']/self.getOption('universalConstants')['PRT']
            self.stateDict['NUTILDA'] = self._spalart(self.stateDict['NU'], self.stateDict['NUT'])
            self.stateDict['K'] = 1.5*np.power(self.stateDict['U']*self.getOption('turbulenceDict')['TIR'], 2)
            self.stateDict['EPSILON'] = 0.09*self.stateDict['RHO']*np.power(self.stateDict['k'], 2)/self.stateDict['RHO']
            self.stateDict['OMEGA'] = self.stateDict['K']/self.stateDict['NUT']
        if mode == 1: # Mach, Reynolds, Altitude
            pass

        #Update dictionary containing boundary conditions
        bcDict = self.getOption('boundaryDict')
        for var in bcDict.keys():
            if bcDict[var]['type']=='volVectorField':
                val = np.array([[self.stateDict[var.upper()], 0.0, 0.0]]) #Set direction in runscript?
            else:
                val = np.array([self.stateDict[var]])
            self.setOption('boundaryDict', {'value': val})

    def _sutherland(self, T):
        MU0 = 1.716e-5
        T0 = 273.15
        C = 110.4
        mu = MU0*np.power(T/T0,1.5)*((T0+C)/(T+C))
        return mu

    def _mach(self, Ma, T):
        specie = self.getOption('thermdynamicsDict')['specie']
        molWeight = self.getOption('speciesDict')[specie]['molWeight']
        dof = self.getOption('speciesDict')[specie]['molDOF']
        gamma = (dof+1)/dof
        RGas = 1e3*self.getOption('universalConstants')['RUniversal']/molWeight #Air Gas Constant
        U = Ma*np.sqrt(gamma*RGas*T)
        return U

    def _reynolds(self, Re, U, D):
        nu = U*D/Re
        return nu

    def _perfectGas(self, rho, T):
        specie = self.getOption('thermdynamicsDict')['specie']
        molWeight = self.getOption('speciesDict')[specie]['molWeight']
        RGas = 1e3*self.getOption('universalConstants')['RUniversal']/molWeight #Air Gas Constant
        p = rho*RGas*T
        return p

    def _spalart(self, nu, nu_t):
        nuTilda = 5*nu
        for _ in range(100):
            x3 = np.power(nuTilda/nu,3)
            nuTilda = nu_t*(x3+np.power(7.1,3))/x3

    def writeOpenFoamHeader(self, file, className, location, objectName):
        """
        Write OpenFOAM header file
        """
        file.write('/*--------------------------------*- C++ -*---------------------------------*\\\n'
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
                   '    class       %s;\n'
                   '    location    \"%s\";\n'
                   '    object      %s;\n'
                   '}\n'
                   '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
                   '\n' % (className, location, objectName))

    def writeControlDict(self):
        """
        Write system/controlDict
        """

        rootDir = self.getOption('dirDict')['rootDir']
        sysDir = self.getOption('dirDict')['sysDir']
        fileName = 'controlDict'
        controlDict = self.getOption('controlDict')

        #Determine parameters from restart code:
        res = controlDict['restart']
        startFrom = 'startTime'
        if res['restart']==1:
            startFrom = 'latestTime'
        startTime = 0
        endTime = controlDict['numIter']
        deltaT = controlDict['deltaT']
        writeInterval = controlDict['printInterval']
        writePrecision = controlDict['precision']
        timePrecision = controlDict['precision']

        filePath = os.path.join(rootDir, sysDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', sysDir, fileName)

        file.write('startFrom       %s;\n'
                   'startTime       %s;\n'
                   'stopAt          endTime;\n'
                   'endTime         %s;\n'
                   'deltaT          %s;\n'
                   'writeControl    timeStep;\n'
                   'writeInterval   %s;\n'
                   'purgeWrite      0;\n'
                   'writeFormat     ascii;\n'
                   'writePrecision  %s;\n'
                   'writeCompression on;\n'
                   'timeFormat      general;\n'
                   'timePrecision   %s;\n'
                   'runTimeModifiable true;\n'
                   '\n'
                   'DebugSwitches\n'
                   '{\n'
                   '    SolverPerformance 0;\n'
                   '}' % (startFrom, startTime, endTime, deltaT, writeInterval, writePrecision, timePrecision))
        file.close()

    def writeFvSchemesDict(self):
        """
        Write system/fvSchemes
        """

        rootDir = self.getOption('dirDict')['rootDir']
        sysDir = self.getOption('dirDict')['sysDir']
        fileName = 'fvSchemes'
        fvSchemes = self.getOption('fvSchemes')

        #Determine parameters from solver
        divSchemes = ('divSchemes\n'
                      '{\n'
                      '    default                                             none;\n'
                      '}\n')
        if self.solver in ['DASimpleFoam']:
            divSchemes = ('divSchemes\n'
                          '{\n'
                          '    default                                             none;\n'
                          '    div(phi,U)                                          bounded Gauss linearUpwindV grad(U);\n'
                          '    div(phi,T)                                          bounded Gauss upwind;\n'
                          '    div(phi,nuTilda)                                    bounded Gauss upwind;\n'
                          '    div(phi,k)                                          bounded Gauss upwind;\n'
                          '    div(phi,omega)                                      bounded Gauss upwind;\n'
                          '    div(phi,epsilon)                                    bounded Gauss upwind;\n'
                          '    div((nuEff*dev2(T(grad(U)))))                       Gauss linear;\n'
                          '    div(pc)                                             bounded Gauss upwind;\n'
                          '}')
        elif self.solver in ['DARhoSimpleFoam']:
            divSchemes = ('divSchemes\n'
                          '{\n'
                          '    default                                             none;\n'
                          '    div(phi,U)                                          bounded Gauss linearUpwindV grad(U);\n'
                          '    div(phi,e)                                          bounded Gauss upwind;\n'
                          '    div(phi,h)                                          bounded Gauss upwind;\n'
                          '    div(pc)                                             bounded Gauss upwind;\n'
                          '    div(((rho*nuEff)*dev2(T(grad(U)))))                 Gauss linear;\n'
                          '    div(phi,nuTilda)                                    bounded Gauss upwind;\n'
                          '    div(phi,k)                                          bounded Gauss upwind;\n'
                          '    div(phi,omega)                                      bounded Gauss upwind;\n'
                          '    div(phi,epsilon)                                    bounded Gauss upwind;\n'
                          '    div(phi,K)                                          bounded Gauss upwind;\n'
                          '    div(phi,Ekp)                                        bounded Gauss upwind;\n'
                          '}')
        elif self.solver in ['DARhoSimpleCFoam']:
            divSchemes = ('divSchemes\n'
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
                          '}')

        filePath = os.path.join(rootDir, sysDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', sysDir, fileName)

        file.write('ddtSchemes\n'
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
        file.write(divSchemes)
        file.close()

    def writeFvSolutionDict(self):
        """
        Write system/fvSolution
        """

        rootDir = self.getOption('dirDict')['rootDir']
        sysDir = self.getOption('dirDict')['sysDir']
        fileName = 'fvSolution'
        fvSolution = self.getOption('fvSolution')

        #Determine relaxation from solver
        relaxationFactors = ('relaxationFactors\n'
                             '{\n'
                             '    fields\n'
                             '    {\n'
                             '        \"(p|p_rgh)\"                         0.30;\n'
                             '    }\n'
                             '    equations\n'
                             '    {\n'
                             '        \"(U|T|e|h|nuTilda|k|epsilon|omega)\" 0.70;\n'
                             '    }\n'
                             '\n'
                             '}\n')
        if self.solver in ['DARhoSimpleFoam']:
            relaxationFactors = ('relaxationFactors\n'
                                 '{\n'
                                 '    fields\n'
                                 '    {\n'
                                 '        \"(p|p_rgh|rho)\"                     0.30;\n'
                                 '    }\n'
                                 '    equations\n'
                                 '    {\n'
                                 '        \"(U|T|e|h|nuTilda|k|epsilon|omega)\" 0.70;\n'
                                 '    }\n'
                                 '\n'
                                 '}\n')
        elif self.solver in ['DARhoSimpleCFoam']:
            relaxationFactors = ('relaxationFactors\n'
                                 '{\n'
                                 '    fields\n'
                                 '    {\n'
                                 '        "(p|rho)"                      1.0;\n'
                                 '    }\n'
                                 '    equations\n'
                                 '    {\n'
                                 '        p                                 1.0;\n'
                                 '        \"(U|T|e|h|nuTilda|k|epsilon|omega)\" 0.80;\n'
                                 '    }\n'
                                 '}')

        filePath = os.path.join(rootDir, sysDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', sysDir, fileName)

        file.write('SIMPLE\n'
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
                   '\n'
                   'potentialFlow\n'
                   '{\n'
                   '    nNonOrthogonalCorrectors           20;\n'
                   '}\n')
        file.write(relaxationFactors)
        file.close()

    def writeDecomposeParDict(self, nProcs):
        """
        Write system/decomposeParDict
        """

        rootDir = self.getOption('rootDir')
        sysDir = self.getOption('sysDir')
        fileName = 'decomposeParDict'
        decompDict = self.getOption('decomposeParDict')

        filePath = os.path.join(rootDir, sysDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', sysDir, fileName)
        
        n = decompDict['simpleCoeffs']['n']
        file.write('numberOfSubdomains     %d;\n'
                   '\n'
                   'method                 %s;\n'
                   '\n'
                   'simpleCoeffs \n'
                   '{ \n'
                   '    n                  (%d %d %d);\n'
                   '    delta              %f;\n'
                   '} \n'
                   '\n'
                   'distributed            false;\n'
                   '\n'
                   'roots();\n' % (nProcs, decompDict['method'], n[0], n[1], n[2], decompDict['simpleCoeffs']['delta']))
        if len(decompDict['preservePatches']) == 1 and decompDict['preservePatches'][0] == 'None':
            pass
        else:
            file.write('\n'
                       'preservePatches        (')
            for pPatch in decompDict['preservePatches']:
                file.write('%s ' % pPatch)
            file.write(');\n')
        if decompDict['singleProcessorFaceSets'][0] != 'None':
            file.write('singleProcessorFaceSets  (')
            for pPatch in decompDict['singleProcessorFaceSets']:
                file.write(' (%s -1) ' % pPatch)
            file.write(');\n')
        file.write('\n'
                   '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        file.close()

    def writeChangeDictionaryDict(self, changeVars=[]):

        rootDir = self.getOption('rootDir')
        sysDir = self.getOption('sysDir')
        fileName = 'changeDictionaryDict'

        filePath = os.path.join(rootDir, sysDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', sysDir, fileName)

        boundaryDict = self.getOption('boundaryDict')
        for var in changeVars:
            boundary = boundaryDict[var]
            file.write('%s\n' % var)
            file.write('{\n')
            file.write('    boundaryField\n')
            file.write('    {\n')
            for boundaryName in self.boundaries.keys():
                boundaryType = self.boundaries[boundaryName]['type']
                if boundaryType in boundary.keys():
                    varType = boundaryDict['type']
                    fixedValue = boundaryDict['value']
                    if varType=='volVectorField':
                        value = '(%s)' % ' '.join([str(_) for _ in fixedValue[0]])
                    elif varType=='volScalarField':
                        value = str(fixedValue[0])

                    boundaryCondition = boundary[boundaryType]
                    if boundaryCondition in ['fixedValue', 'nutUSpaldingWallFunction', 'alphatWallFunction',
                                             'kqRWallFunction', 'epsilonWallFunction', 'omegaWallFunction']:
                        file.write('        %s\n' % boundaryName)
                        file.write('        {\n')
                        file.write('            value        uniform %s;\n' % value)
                        file.write('        }\n')
                    elif boundaryCondition in ['inletOutlet']:
                        file.write('        %s\n' % boundaryName)
                        file.write('        {\n')
                        file.write('            inletValue   uniform %s;\n' % value)
                        file.write('            value        uniform %s;\n' % value)
                        file.write('        }\n')
            file.write('    }\n')
            file.write('}\n\n')
        file.close()

    def writeBlockMeshDict(self):
        pass

    def writeSurfaceFeatureExtractDict(self):
        pass

    def writeSnappyHexMeshDict(self):
        pass

    def writeTransportProperties(self):
        """
        Writes constant/transportProperties
        """
        
        rootDir = self.getOption('dirDict')['rootDir']
        constDir = self.getOption('dirDict')['constDir']
        fileName = 'transportProperties'

        #Determine parameters
        model = self.getOption('thermodynamicsDict')['transportModel']
        kinematicViscosity = self.stateDict['NU']
        prandtl = self.getOption('universalConstants')['Pr']
        turbulentPrandtl = self.getOption('universalConstants')['Prt']

        filePath = os.path.join(rootDir, constDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', constDir, fileName)

        file.write('transportModel %s;\n'
                   '\n'
                   'nu %f;\n'
                   'Pr %f;\n'
                   'Prt %f;' % (model, kinematicViscosity, prandtl, turbulentPrandtl))
        file.close()

    def writeThermophysicalProperties(self):
        """
        Writes constant/thermophysicalProperties
        """
        
        rootDir = self.getOption('dirDict')['rootDir']
        constDir = self.getOption('dirDict')['constDir']
        fileName = 'thermophysicalProperties'

        #Determine parameters
        equationOfState = self.getOption('thermdynamicsDict')['equationOfState']
        thermoModel = self.getOption('thermdynamicsDict')['model']
        thermoType = self.getOption('thermodynamicsDict')['type']
        transportModel = self.getOption('thermdynamicsDict')['transport']
        specie = self.getOption('thermdynamicsDict')['specie']
        molWeight = self.getOption('speciesDict')[specie]['molWeight']
        dof = self.getOption('speciesDict')[specie]['molDOF']
        Cp = 0.5*(1.0+dof)*self.getOption('universalConstants')['RUniversal']
        Hf = self.getOption('speciesDict')[specie]['Hf']
        dynamicViscosity = self.stateDict['MU']
        prandtl = self.getOption('universalConstants')['Pr']
        TRef = self.getOption('universalConstants')['TRef']

        filePath = os.path.join(rootDir, constDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', constDir, fileName)

        file.write('thermoType \n'
                   '{\n'
                   '    mixture               pureMixture;\n'
                   '    specie                specie;\n'
                   '    equationOfState       %s;\n'
                   '    energy                sensibleInternalEnergy;\n'
                   '    thermo                %s\n'
                   '    type                  %s;\n'
                   '    transport             %s;\n'
                   '}\n'
                   '\n'
                   'mixture \n'
                   '{\n'
                   '    specie\n'
                   '    {\n'
                   '        molWeight           %f;\n'
                   '    }\n'
                   '    thermodynamics\n'
                   '    {\n'
                   '        Cp                  %f;\n'
                   '        Hf                  %f;\n'
                   '    }\n' % (equationOfState, thermoModel, thermoType, transportModel, molWeight, Cp, Hf))
        if transportModel in ['const']:
            file.write('    transport\n'
                       '    {\n'
                       '        mu                  %f;\n'
                       '        Pr                  %f;\n'
                       '        TRef                %f;\n'
                       '    }\n'
                       '}\n' % (dynamicViscosity, prandtl, TRef))
        elif transportModel in ['sutherland']:
            pass
        file.close()

    def writeTurbulenceProperties(self):
        """
        Writes constant/turbulenceProperties
        """
        
        rootDir = self.getOption('dirDict')['rootDir']
        constDir = self.getOption('dirDict')['constDir']
        fileName = 'turbulenceProperties'

        #Determine parameters
        model = self.getOption('turbulenceProperties')['model']
        turbulentPrandtl = self.getOption('universalConstants')['Prt']

        filePath = os.path.join(rootDir, constDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, 'dictionary', constDir, fileName)

        file.write('simulationType RAS;\n'
                   'RAS \n'
                   '{ \n'
                   '    RASModel             %s;\n'
                   '    turbulence           on;\n'
                   '    printCoeffs          off;\n'
                   '    nuTildaMin           1e-16;\n'
                   '    Prt                  %f;\n'
                   '}' % (model, turbulentPrandtl))
        file.close()

    def writeBC(self, varName):
        """
        Writes boundary condition file
        """

        rootDir = self.getOption('dirDict')['rootDir']
        bcDir = self.getOption('dirDict')['bcDir']
        fileName = varName

        #Determine parameters
        boundaryDict = self.getOption('boundaryDict')[varName]
        varType = boundaryDict['type']
        varDim = boundaryDict['dimensions']
        internalField = 'internalField'
        fixedValue = boundaryDict['value']
        if varName=='p' and self.solver=='DASimpleFoam':
            varDim = [0, 2, -2, 0, 0, 0, 0]
            fixedValue = np.array([0.0])
        if varType=='volVectorField':
            internalValue = '(%s)' % ' '.join([str(_) for _ in fixedValue[0]])
        elif varType=='volScalarField':
            internalValue = str(fixedValue[0])
        else:
            raise Exception('Unhandled combination of variable type and values')

        filePath = os.path.join(rootDir, bcDir, fileName)
        file = open(filePath, 'w')
        self.writeOpenFoamHeader(file, varType, bcDir, fileName)

        file.write('dimensions      [%s];\n\n' % ' '.join([str(_) for _ in varDim]))
        file.write('%s   uniform %s;\n\n' % (internalField, internalValue))
        file.write('boundaryField\n'
                   '{\n')
        for boundaryName in self.boundaries.keys():
            boundaryType = self.boundaries[boundaryName]['type']
            if boundaryType in ['symmetry', 'symmetryPlane']:
                boundaryCondition = boundaryType
            else:
                boundaryCondition = boundaryDict[boundaryType]
            file.write('    %s\n'
                       '    {\n' % (boundaryName))
            if boundaryCondition in ['fixedZero']:
                value = '0.0' if varType=='volScalarField' else '(0.0 0.0 0.0)'
                file.write('        type            fixedValue;\n'
                           '        value           %s;\n'
                           '   }\n' % value)
            elif boundaryCondition in ['inletOutlet']:
                file.write('        type            inletOutlet;\n'
                           '        inletValue      $%s;\n'
                           '        value           $%s;\n'
                           '   }\n' % (internalField, internalField))
            elif boundaryCondition in ['calculated', 'fixedValue', 'nutUSpaldingWallFunction', 'alphatWallFunction',
                                       'kqRWallFunction', 'epsilonWallFunction']:
                boundaryCondition = 'compressible::alphatWallFunction' if boundaryCondition=='alphatWallFunction' else boundaryCondition
                file.write('        type            %s;\n'
                           '        value           $%s;\n'
                           '   }\n' % (boundaryCondition, internalField))
            elif boundaryCondition in ['omegaWallFunction']:
                file.write('        type            %s;\n'
                           '        value           $%s;\n'
                           '        blended         true;\n'
                           '   }\n' % (boundaryCondition, internalField))
            elif boundaryCondition in ['symmetry', 'symmetryPlane', 'zeroGradient']:
                file.write('        type            %s;\n'
                           '   }\n' % (boundaryCondition))
            else:
                raise Exception(boundaryCondition + ' boundary condition unhandled')
        file.write('}')
        file.close()

    def writeParaview(self):
        rootDir = self.getOption('dirDict')['rootDir']
        fileName = 'paraview.foam'
        filePath = os.path.join(rootDir, fileName)
        file = open(filePath,'w')
        file.close()

    def readBoundaries(self, boundaryFile):
        if (os.path.exists(boundaryFile)):
            file = open(boundaryFile, 'r')
            lines = file.readlines()

            begin = True
            patchName = False
            boundaryDict = {}
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
                            boundaryDict[name] = {'type': splitLine[1][:-1]}
                        elif splitLine[0]=='}':
                            patchName = True
            file.close()
            self.boundaries = boundaryDict
            return boundaryDict
        else:
            raise Exception(boundaryFile + ' does not exist')

    def getOption(self, name):
        """
        Get a value from options

        Parameters
        ----------
        name : str
           Name of option to get. Not case sensitive

        Returns
        -------
        value : varies
           Return the value of the option.
        """

        if name in self.defaultOptions:
            return self.options[name][1]
        else:
            raise Exception('%s is not a valid option name.' % name)

    def setOption(self, name, value):
        """
        Set a value to options.
        See pyDAFoam.py setOption
        """

        try:
            self.defaultOptions[name]
        except KeyError:
            Exception('Option \"%-30s\" is not a valid %s option.' % (name, self.name))

        # Make sure we are not trying to change an immutable option if
        # we are not allowed to.
        if name in self.imOptions:
            raise Exception('Option \"%-35s\" cannot be modified after the solver is created.' % name)

        # Now we know the option exists, lets check if the type is ok:
        if isinstance(value, self.defaultOptions[name][0]):
            # the type matches, now we need to check if the 'value' is of dict type, if yes, we only
            # replace the subKey values of 'value', instead of overiding all the subKey values
            # NOTE. we only check 3 levels of subKeys
            if isinstance(value, dict):
                for subKey1 in value:
                    # check if this subKey is still a dict.
                    if isinstance(value[subKey1], dict):
                        for subKey2 in value[subKey1]:
                            # check if this subKey is still a dict.
                            if isinstance(value[subKey1][subKey2], dict):
                                for subKey3 in value[subKey1][subKey2]:
                                    self.options[name][1][subKey1][subKey2][subKey3] = value[subKey1][subKey2][subKey3]
                            else:
                                self.options[name][1][subKey1][subKey2] = value[subKey1][subKey2]
                    else:
                        # no need to set self.options[name][0] since it has the right type
                        self.options[name][1][subKey1] = value[subKey1]
            else:
                # It is not dict, just set
                # no need to set self.options[name][0] since it has the right type
                self.options[name][1] = value
        else:
            raise Exception(
                'Datatype for Option %-35s was not valid\n'
                'Expected data type is %-47s\n'
                'Received data type is %-47s' % (name, self.defaultOptions[name][0], type(value))
            )

    def updateFOAMOption(self):
        pass

    #Below only used internally

    def _initializeOptions(self, options):
        #Make sure options were passed in
        if options is None:
            raise Exception('The \"options\" keyword argument must be passed to AutoFOAM.')

        # set immutable options that users should not change during the optimization
        self.imOptions = self._getImmutableOptions()

        # Load all the default option information:
        self.defaultOptions = self._getDefOptions()

        # Set options based on defaultOptions
        # we basically overwrite defaultOptions with the given options
        # first assign self.defaultOptions to self.options
        self.options = OrderedDict()
        for key in self.defaultOptions:
            if len(self.defaultOptions[key]) != 2:
                raise Exception(
                    'key %s has wrong format!\n'
                    'Example: {\"iters\" : [int, 1]}\n' % key
                )
            self.options[key] = self.defaultOptions[key]
        # now set options to self.options
        for key in options:
            self._initOption(key, options[key])
        return

    def _getImmutableOptions(self):
        """
        We define the list of options that *cannot* be changed after the
        object is created. pyDAFoam will raise an error if a user tries to
        change these. The strings for these options are placed in a set
        """
        return ('universalConstants', 'speciesDict')

    def _getDefOptions(self):
        """
        Setup default options

        Returns
        -------

        defOpts : dict
            All the OpenFOAM options.
        """

        # initialize the DAOPTION object
        daOption = FOAMOPTION()

        defOpts = {}

        # assign all the attribute of daOptoin to defOpts
        for key in vars(daOption):
            value = getattr(daOption, key)
            defOpts[key] = [type(value), value]

        return defOpts

    def _initOption(self, name, value):
        """
        Set a value to options. This function will be used only for initializing the options internally.

        Parameters
        ----------
        name : str
           Name of option to set. Not case sensitive
        value : varies
           Value to set. Type is checked for consistency.
        """

        try:
            self.defaultOptions[name]
        except KeyError:
            Exception('Option \"%-30s\" is not a valid %s option.' % (name, self.name))

        # Make sure we are not trying to change an immutable option if
        # we are not allowed to.
        if name in self.imOptions:
            raise Exception('Option \"%-35s\" cannot be modified after the solver is created.' % name)

        # Now we know the option exists, lets check if the type is ok:
        if isinstance(value, self.defaultOptions[name][0]):
            # the type matches, now we need to check if the 'value' is of dict type, if yes, we only
            # replace the subKey values of 'value', instead of overiding all the subKey values
            if isinstance(value, dict):
                for subKey in value:
                    # no need to set self.options[name][0] since it has the right type
                    self.options[name][1][subKey] = value[subKey]
            else:
                # It is not dict, just set
                # no need to set self.options[name][0] since it has the right type
                self.options[name][1] = value
        else:
            raise Exception(
                'Datatype for Option %-35s was not valid\n'
                'Expected data type is %-47s\n'
                'Received data type is %-47s' % (name, self.defaultOptions[name][0], type(value)))