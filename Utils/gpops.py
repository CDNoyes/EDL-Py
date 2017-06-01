import matlab
import matlab.engine 

doc_path = "C:\Users\cdnoyes\Documents"
gpops_path = "C:\Users\cdnoyes\Documents\MATLAB\GPOPS-II"

def gpops(input):
    """ Calls GPOPS-II with the given input structure (dict in python) 
    
        By default, the code will attempt to attach to any existing Matlab instance,
        which is much faster than opening a new one. In order to connect to an existing 
        instance, you must call "matlab.engine.shareEngine" in Matlab. 
    
    """

    try:
        eng = matlab.engine.connect_matlab()
        print "Connected to existing (shared) Matlab instance."
    except:
        print "No current Matlab instance found, creating a new instance..."
        eng = matlab.engine.start_matlab()
        print "Matlab instance created successfully."
        eng.addpath(gpops_path, nargout=0)
        print eng.pwd()
        eng.gpopsMatlabPathSetup
        print "gpops-II paths set..."
    
    # Call a test example - succeeds 
    # eng.cd(gpops_path+"\\examples\\brachistochone", nargout=0)
    # eng.brachistochroneMain(nargout=0)
    
    
    # Call my own srp optimization
    eng.cd(doc_path+"\\PDOpt")
    sol = eng.getOptTrajAccel()
    print sol.keys()
    print sol['phase']['time']
    # eng.gpops(input)
    
    
    
def gpops_input():

    setup = dict()
    functions = dict()
    mesh = ()
    nlp = dict()
    derivatives = dict() 
    
    setup['name'] = 'SRP Optimization'
    functions['continuous'] = 'Dynamics' # These are normally function handles in Matlab, this may not work. A solution is to write matlab functions that take IC and other info and generate the setup in Matlab 
    functions['endpoint'] = 'Cost'
    setup['functions'] = functions
    
    nlp['solver'] = 'snopt';
    nlp['snoptoptions'] = {'tolerance' : 1e-6, 'maxiterations' : 2000 }
    setup['nlp'] = nlp 
    
    # setup['bounds'] = bounds
    # setup['guess'] = guess
    
    # setup['derivatives.supplier'] = 'sparseFD'
    # setup['derivatives.derivativelevel'] = 'first' 
    # setup['derivatives.dependencies'] = 'sparseNaN'
    # setup['scales.method'] = 'automatic-guessUpdate'; # 'none' or 'automatic-bounds' or 'automatic-guess' or 'automatic-guessUpdate' or 'automatic-hybrid' or 'automatic-hybridUpdate' or 'defined'
    # setup['mesh.method'] = 'hp-PattersonRao'; # 'hp-PattersonRao' or 'hp-DarbyRao' or 'hp-LiuRao'
    # setup['mesh.tolerance'] = 1e-3 # Default 1e-3
    # setup['mesh.maxiterations'] = 50 # Default 10
    
    # setup['method'] = 'RPM-Differentiation'
    # setup['displaylevel'] = 0
    return setup 
    
    
if __name__ == "__main__":
    gpops(0)