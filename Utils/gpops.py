import matlab
import matlab.engine

# doc_path = "C:\Users\cdnoyes\Documents"
# gpops_path = "C:\Users\cdnoyes\Documents\MATLAB\GPOPS-II"
doc_path = "E:\Documents"
gpops_path = "E:\Documents\GPOPS-II"

def EG(inputs):
    """ Calls GPOPS-II with the given input structure (dict in python)

        By default, the code will attempt to attach to any existing Matlab instance,
        which is much faster than opening a new one. In order to connect to an existing
        instance, you must call "matlab.engine.shareEngine" in Matlab.

    """

    try:
        engine = matlab.engine.connect_matlab()
        # print("Connected to existing (shared) Matlab instance.")
    except:
        print("No shared Matlab instance found, creating a new instance...")
        engine = matlab.engine.start_matlab()
        print("Matlab instance created successfully.")


    traj = engine.optimal_entry_guidance(*inputs)
    return traj


def entry(inputs):
    """ Calls GPOPS-II with the given input structure (dict in python)

        By default, the code will attempt to attach to any existing Matlab instance,
        which is much faster than opening a new one. In order to connect to an existing
        instance, you must call "matlab.engine.shareEngine" in Matlab.

    """

    try:
        engine = matlab.engine.connect_matlab()
        # print("Connected to existing (shared) Matlab instance.")
    except:
        print("No shared Matlab instance found, creating a new instance...")
        engine = matlab.engine.start_matlab()
        print("Matlab instance created successfully.")


    traj = engine.optimize_entry(*inputs)
    return traj


def srp(inputs):
    """ Calls GPOPS-II to solve planetary landing problem

        By default, the code will attempt to attach to any existing Matlab instance,
        which is much faster than opening a new one. In order to connect to an existing
        instance, you must call "matlab.engine.shareEngine" in Matlab.

    """

    try:
        engine = matlab.engine.connect_matlab()
        # print("Connected to existing (shared) Matlab instance.")
    except:
        print("No shared Matlab instance found, creating a new instance...")
        engine = matlab.engine.start_matlab()
        print("Matlab instance created successfully.")


    traj = engine.optimize_srp(*inputs)
    return traj


if __name__ == "__main__":
    pass
