"""implement base simulation characteristics"""
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Imports                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from datetime import datetime
from os.path import isdir
import logging
from kgsim.exceptions import SimulationNotFoundError
from kbasic.environment import simulationDir
from kbasic.parsing import Folder, File, ensure_path, configure_log
from kbasic.user_input import yesno
from matplotlib.pyplot import cm as cmaps

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                   Functions                                    <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def locate_sim(path: str) -> Folder:
    """
    find a simulation
    
    Args
    ----
    path: str - the location of the sim you want to find

    Returns
    -------
    dr: Folder - a pathlib-esque folder containing the simulation

    Raises
    ------
    SimulationNotFoundError - if can't find path
    """
    dr = Folder(path)
    if dr.exists: return dr
    # else check the default sim path
    dr = simulationDir / path
    if dr.exists: return dr
    raise SimulationNotFoundError(f"could not find {path=} or {(simulationDir / path)=}")

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Classes                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class GenericSimulation:
    def __init__(
            self,
            path: str,
            template: str|Folder= None,
            caching: bool= False,
            verbose: bool= False,
            debug: bool = False,
        ) -> None:
        """A generic class for all simulations

        Args:
            path (str): the location of the simulation
            template (str | Folder, optional): the location of the template this simulation is 
            based on. Defaults to None.
            caching (bool, optional): whether or not to set up a cache to store data in. 
            Defaults to False.
            verbose (bool, optional): whether to print a bunch of bullshit. Defaults to True.
        """
        try:
            self.dir = locate_sim(path)
            self.path = self.dir.path
            self.name = self.dir.name
        except SimulationNotFoundError:
            if template and yesno(f"""
            Can't find {path}...
            would you like to create this simulation from template: {template}?
            """):
                template.copy(path)
                self.dir = Folder(path)
                self.path = self.dir.path
                self.name = self.dir.name
        #setup log
        ensure_path(f"{self.path}/log")
        self.log = configure_log(
            f"{__package__}.{self.__class__.__name__}.{self.name}",
            level=logging.DEBUG if debug else logging.INFO,
            file=f"{self.path}/log/{datetime.now().isoformat()}.log",
            file_level=logging.DEBUG if debug else logging.INFO,
            console_level=logging.DEBUG if debug else logging.INFO if verbose else None
        )
        self.log.info(f"initializing simulation -> {self.name}")
        self.template = template
        self.log.info(f"from template -> {template.name}")
        self.verbose = verbose
        self.debug = debug
        #setup cache
        self.log.info("caching is ON..." if caching else "caching is OFF...")
        self.caching = caching
        self.cache: dict = {}

class SimulationGroup(Folder):
    def __init__(self, path: str, simtype = GenericSimulation, **sim_kwds) -> None:
        Folder.__init__(self, path)
        self.simulations = {x.path.split('/')[-1]:simtype(x, **sim_kwds) for x in self.children \
            if isdir(x) and File(x+"/input/input").exists}
    def __repr__(self) -> str:
        return self.name+'\n'+'-'*20+"\n"+"\n".join([
            f"{k}: {repr(v)}" for k, v in self.simulations.items()
        ])
    def __len__(self) -> int: return len(self.simulations)
    def __getitem__(self, item): return self.simulations[item]
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index < len(self):
            i = self.index
            self.index += 1
            return list(self.values())[i]
        else: raise StopIteration

    def sort_by(self, key: str) -> None:
        """docstring"""
        new_simulations = {
            v.__dict__[key]: v \
            for k, v in sorted(
                self.simulations.items(), key = lambda item: item[1].__dict__[key]
            )
        }
        if len(new_simulations) != len(self.simulations):
            raise KeyError(f"""
                the simulation value: {key} is not unique in {self.name}, please provide a
                 unique key to sort by.
            """)
        self.simulations = new_simulations

    def colorer(self, cmap=cmaps.plasma) -> list:
        """docstring"""
        return [cmap(i / (len(self)+.1)) for i in range(len(self))]
    def labeler(self) -> list[str]:
        """docstring"""
        return [x.name for x in self.simulations.values()]
    def plotter(self, cmap=cmaps.plasma) -> list[tuple]:
        """docstring"""
        return [
            (l_i, c_i, sim_i) \
            for l_i, c_i, sim_i in zip(
                self.labeler(), self.colorer(cmap=cmap), self.simulations.values()
            )
        ]
    def items(self):
        """docstring"""
        return self.simulations.items()
    def values(self):
        """docstring"""
        return self.simulations.values()
    def keys(self):
        """docstring"""
        return self.simulations.keys()
