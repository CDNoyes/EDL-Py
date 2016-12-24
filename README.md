# EDL-Py
Research-grade Entry, Descent, and Landing software in Python 2.7
![alt text](https://github.com/CDNoyes/EDL-Py/blob/master/SimulationFSM.gif "A trajectory state machine graph")


## Dependencies
- Standard numerical computing packages (numpy, scipy, pandas, etc.)
- [transitions](https://github.com/tyarkoni/transitions) - FSM software
- [chaospy](https://github.com/hplgit/chaospy) - software for polynomial chaos expansions and design of experiments
- [Numdifftools](https://pypi.python.org/pypi/Numdifftools) - numerical differentiation methods

### Optional
- pygraphviz - FSM visualization, dependent on graphviz and swig. Difficult to install on Windows, I highly recommend using [these wheels with pip.](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz)

