# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.fields import ScalarField, VectorField
from kgsim.simulation import GenericSimulation, SimulationGroup
from kgsim.particles import Species, Particle
from kgsim.dhybridr.io import dHybridRinput, dHybridRout
from kgsim.dhybridr.initializer import dHybridRinitializer
from kgsim.dhybridr.anvil_submit import AnvilSubmitScript
from kgsim.templates import dHybridRtemplate

from kplot import show, func_video
from kbasic.parsing import Folder
from kbasic.user_input import yesno
from os import system
import numpy as np 
from h5py import File as h5File

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
track_keys = ['B1', 'B2', 'B3', 'E1', 'E2', 'E3', 'ene', 'n', 'p1', 'p2', 'p3', 'q', 't', 'x1', 'x2', 'x3']

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# simulation parsing
def extract_energy(file_name: str) -> tuple:
    with h5File(file_name, 'r') as file:
        fE = np.mean(file["DATA"], axis=1)
        [low, high] = file["AXIS"]["X2 AXIS"][:]
        lne = np.linspace(low, high, fE.shape[0])
        dlne = np.diff(lne)[0]
        E = np.exp(lne)
        return E, fE, dlne
def iters(simulation) -> list[int]:
    """Grab the iterations of each snapshot for a simulation"""
    return [int(fn[-11:-3]) for fn in simulation.density.file_names]
def times(simulation: GenericSimulation, ndigits: int = 7) -> list[float]:
    """return the time (in simulation units) of each measurement

    Args:
        simulation (GenericSimulation): the sim you want to extract times from
        ndigits (int, optional): how many digits to round to (helps with floating point error). Defaults to 7.

    Returns:
        list[float]: the time (in simulation units) of each snapshot of the simulation.
    """
    x = list(np.array(iters(simulation)).astype(float) * simulation.dt)
    return [round(xi, ndigits) for xi in x]
def particle_video(
    sim,
    particles: list[str] | int,
    fname: str,
    background = None,      
    res: int = None,
    zfill: int = 10,
    dpi: int = 250,
    # paticle plotting keywords
    color = 'red',
    marker='.',
    ms=10, 
    # assume the rest are keywords for the show function
    **kwds
):
    # check species
    assert sim.input.num_species == 1, "only one species is implemented rn :-("
    # check the time resolution is valid
    if not res: res = sim.input.sp01.track_nstore
    dnp = sim.input.sp01.track_nstore
    dnb = sim.input.ndump
    assert (dnp % res == 0) & (res % dnb == 0), "your resolution must be an integer multiple of track_nstore and ndump must be an integer multiple of res."
    # find the file
    fn = sim.path+"/Output/Tracks/Sp01/track_Sp01.h5"
    with h5File(fn) as file:
        # extract particle tracks based on the particles argument
        tags = np.array(list(file.keys()))
        match particles:
            case [str(x), ]: selected = particles 
            case int(x): selected = np.random.choice(tags, size=x, replace=False)
        sx, sy = np.array([file[t]['x1'] for t in selected]).T, np.array([file[t]['x2'] for t in selected]).T 
        # setup background
        if not background: background = sim.density
        # initialize plots
        fig,ax,img = show(
            background[0], 
            x=range(0, sim.input.boxsize[0], sim.dx), y=range(0, sim.input.boxsize[1], sim.dy), 
            zorder=1,
            **kwds
        )
        line, = ax.plot(
            sx[0], sy[0], 
            ls='None', color=color, marker=marker, ms=ms,
            zorder = 2
        )
        # execute the loop
        def update(i: int, dnp=dnp, dnb=dnb):
            pind = i * dnp
            line.set_data(sx[pind], sy[pind])
            if pind % dnb == 0:
                bind = pind // dnb 
                img.set_array(background[bind])
        func_video(fname, fig, update, )

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class dHybridRparticle(Particle):
    def __init__(
        self, 
        species: Species,
        tag: str,  
        load: bool = True
        ) -> None:
        """docstring"""
        self.tag = tag
        self.species = self.sp = species
        Particle.__init__(self, species, tag)
        self.loaded: bool = False
        if load: self.load()
    def __len__(self) -> int: 
        assert self.loaded, f"Tried to take the length of {self} without loading first"
        return len(self.x)
    def load(self):
        with h5File(self.species.path) as file:
            for k in track_keys:
                try: setattr(self, k, file[self.tag][k][:])
                except: continue
            self.x = self.x1 
            self.y = self.x2
            self.loaded: bool = True
class dHybridRspecies(Species):
    def __init__(
        self, 
        parent: GenericSimulation, 
        species_number: int
        ) -> None:
        """docstring"""
        self.n = self.number = self.sp_num = species_number
        self.nstr = str(self.n).zfill(2)
        Species.__init__(self, f"sp{self.nstr}")
        self.parent = parent 
        self.input = getattr(parent.input, self.name)
        self.pipsi = parent.input.ndump / parent.input.sp01.track_nstore # particle indices per sim index
        self.loaded = False
        if self.input.track_dump:
            self.path = parent.path+f"/Output/Tracks/Sp{self.nstr}/track_Sp{self.nstr}.h5"
            self.load()
    def __repr__(self) -> str: return self.name
    def __str__(self) -> str: return self.name
    def __len__(self) -> int: 
        assert self.loaded, f"Tried to take the length of {self} without loading first"
        return len(self.tags)
    def __getitem__(self, key):
        match key:
            case str(tag): return dHybridRparticle(self, tag)
            case int(ind): return dHybridRparticle(self, self.tags[ind])
            case slice(start=i, stop=j, step=di): 
                return [
                    dHybridRparticle(self, self.tags[k]) for k in range(
                        0 if i is None else i, 
                        len(self.tags) if j is None else j, 
                        1 if di is None else di
                        )
                    ]
    def load(self):
        with h5File(self.path) as file:
            self.tags = np.array(list(file.keys()))
            self.loaded = True 

    def sample(self, N: int = 1, load: bool = True): 
        particle_list = np.array([
            dHybridRparticle(self, t, load=load) for t in np.random.choice(self.tags, size=N)
            ])
        if len(particle_list)==1: return particle_list[0]
        else: return particle_list
class dHybridR(GenericSimulation):
    """
    A simulation class to interact with dHybridR simulations in python
    """
    def __init__(
            self, 
            path: str,
            caching: bool = False,
            verbose: bool = False,
            template: Folder = dHybridRtemplate,
            compressed: bool = False
        ) -> None:
        self.compressed = compressed
        #setup simulation
        GenericSimulation.__init__(self, path, caching=caching, verbose=verbose, template=template)
        #setup input, output, and restart folders
        self.parse_input()
        self.output = Folder(self.path+"/Output")
        if not self.output.exists and verbose:
            if yesno("There is no output, would you like to run this simulation?\n"): 
                self.run()
        elif self.output.exists: 
            self.parse_output()
            self.ncores: int = int(np.prod(self.input.node_number))
            self.ncores_charged: int = self.ncores + self.ncores % 128
            if self.out.exists:
                self.runtime: float = self.out.runtime #run time as calculated from out file, in hours
                self.corehours: float = self.runtime * np.prod(self.ncores)
                self.corehours_charged: float = self.runtime * self.ncores_charged
        self.restartDir = Folder(self.path+"/Restart")
    def __repr__(self) -> str: return self.name
    def __len__(self) -> int:
        if self.output.exists: 
            return len(self.density)
        else: return 0
    def create(self) -> None:
        self.template.copy(self.path)
        system(f"chmod 755 {self.path}/dHybridR")
    def parse_input(self) -> None:
        self.input = dHybridRinput(self.path+"/input/input")
        self.dt = self.input.dt
        self.niter = self.input.niter
        self.dx: float = self.input.boxsize[0]/self.input.ncells[0]
        self.dy: float = self.input.boxsize[1]/self.input.ncells[1]
        self.dz: float = self.input.boxsize[2]/self.input.ncells[2] if len(self.input.boxsize)==3 else np.inf
    def run(self, initializer: dHybridRinitializer, submit_script: AnvilSubmitScript) -> None:
        initializer.prepare_simulation()
        submit_script.write()
        system(f"sh {submit_script.path}")
    def parse_output(self) -> None:
        self.out = dHybridRout(self.path+"/out")
        kwargs = {'caching':self.caching, 'verbose':self.verbose, 'parent':self}
        self.B       = VectorField(self.path + "/Output/Fields/Magnetic/Total/", name="magnetic", latex="B", **kwargs)
        self.E       = VectorField(self.path + "/Output/Fields/Electric/Total/", name="electric", latex="E", **kwargs)
        self.etx1    = ScalarField(self.path + "/Output/Phase/etx1/Sp01/", **kwargs)
        self.pxx1    = ScalarField(self.path + "/Output/Phase/p1x1/Sp01/", **kwargs)
        self.pyx1    = ScalarField(self.path + "/Output/Phase/p2x1/Sp01/", **kwargs)
        self.pzx1    = ScalarField(self.path + "/Output/Phase/p3x1/Sp01/", **kwargs)
        self.density = ScalarField(self.path + "/Output/Phase/x3x2x1/Sp01/", name="density", latex=r"$\rho$", **kwargs)
        self.Pxx     = ScalarField(self.path + "/Output/Phase/PressureTen/Sp01/xx/", **kwargs)
        self.Pyy     = ScalarField(self.path + "/Output/Phase/PressureTen/Sp01/yy/", **kwargs)
        self.Pzz     = ScalarField(self.path + "/Output/Phase/PressureTen/Sp01/zz/", **kwargs)
        self.u       = VectorField(self.path + "/Output/Phase/FluidVel/Sp01/", name="bulkflow", latex="u", **kwargs)
        [
            self.energy_grid,
            self.energy_pdf,
            self.dlne
        ] = np.array([[*extract_energy(f)] for f in self.etx1.file_names], dtype=object).T
        self.energy_grid = np.vstack(self.energy_grid) 
        self.energy_pdf = np.vstack(self.energy_pdf) 
        self.dlne = np.vstack(self.dlne) 
        if self.input.sp01.track_dump:
            self.sp01 = dHybridRspecies(self, 1)
        self.iter = np.array(iters(self))
        self.time = np.array(times(self))
class dHybridRgroup(SimulationGroup):
    def __init__(self, path, **sim_kwds):
        SimulationGroup.__init__(self, path, simtype=dHybridR, **sim_kwds)