# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from pysim.utils import yesno
from pysim.parsing import Folder, File
from pysim.environment import simulationDir

from os.path import isdir

from matplotlib.pyplot import cm as cmaps

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class GenericSimulation:
    def __init__(
            self, 
            path:str, 
            template:str|Folder=None,
            caching:bool=False,
            verbose:bool=True,
        ) -> None:
        """A generic class for all simulations

        Args:
            path (str): the location of the simulation
            template (str | Folder, optional): the location of the template this simulation is based on. Defaults to None.
            caching (bool, optional): whether or not to set up a cache to store data in. Defaults to False.
            verbose (bool, optional): whether to print a bunch of bullshit. Defaults to True.
        """
        self.template = template
        self.verbose = verbose
        #setup cache
        if self.verbose: print("caching is ON..." if caching else "caching is OFF...")
        self.caching = caching 
        self.cache: dict = {}
        #make sure the simulation exists
        if verbose: print(f"Finding path: {path}")
        self.path: str = path
        self.dir = Folder(path)
        self.name = self.dir.name
        #if the given path doesn't exist, check the default simulation directory 
        if not self.dir.exists: 
            if self.verbose: print(f"No simulation found in {path}, checking default simulation directory: {simulationDir.path}...")
            default_path: str = f"{simulationDir.path}/{self.name}"
            self.path = default_path
            self.dir = Folder(self.path)
            #if simulation isn't in default simulation directory either copy a template to that location or raise an error
            if not self.dir.exists:
                if yesno(f"No such simulation exists, would you like to copy \ntemplate: {template.name}, \nto location: {self.path}?\n"):
                    self.create()
                else: raise FileNotFoundError("Please create simulation and try again")
    
    def create(self):
        self.template.copy(self.path)

class SimulationGroup(Folder):
    def __init__(self, path: str, simtype = GenericSimulation, **sim_kwds) -> None: 
        Folder.__init__(self, path)
        self.simulations = {x.split('/')[-1]:simtype(x, **sim_kwds) for x in self.children if isdir(x) and File(x+"/input/input").exists}
    
    def __repr__(self) -> str: return self.name+'\n'+'-'*20+"\n"+"\n".join([f"{k}: {repr(v)}" for k, v in self.simulations.items()])
    def __len__(self) -> int: return len(self.simulations)
    def __getitem__(self, item): return self.simulations[item]

    def sort_by(self, key: str) -> None:
        new_simulations = {v.__dict__[key]: v for k, v in sorted(self.simulations.items(), key = lambda item: item[1].__dict__[key])}
        if len(new_simulations) != len(self.simulations): raise KeyError(f"the simulation value: {key} is not unique in {self.name}, please provide a unique key to sort by.")
        self.simulations = new_simulations

    def colorer(self, cmap=cmaps.plasma) -> list: return [cmap(i / (len(self)+.1)) for i in range(len(self))]
    def labeler(self) -> list[str]: return [x.name for x in self.simulations.values()]
    def plotter(self, cmap=cmaps.plasma) -> list[tuple]: return [(l_i, c_i, sim_i) for l_i, c_i, sim_i in zip(self.labeler(), self.colorer(cmap=cmap), self.simulations.values())]