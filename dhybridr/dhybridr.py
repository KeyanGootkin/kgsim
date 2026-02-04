# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
#pysim imports
from pysim.utils import yesno 
from pysim.parsing import Folder
from pysim.environment import dHybridRtemplate
from pysim.fields import ScalarField, VectorField
from pysim.simulation import GenericSimulation
from pysim.particles import Species, Particle
from pysim.plotting import show
from pysim.dhybridr.io import dHybridRinput, dHybridRout
from pysim.dhybridr.initializer import dHybridRinitializer, TurbInit, dHybridRconfig
from pysim.dhybridr.anvil_submit import AnvilSubmitScript
#nonpysim imports
import numpy as np 
from h5py import File as h5File
from os import system
from glob import glob

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

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class dHybridRparticle(Particle):
    def __init__(self, tag: str, species: Species, load: bool = False):
        self.tag = tag
        self.species = self.sp = species
        if load: self.load()
    def load(self):
        with h5File(self.species.path) as file:
            for k in track_keys:
                try: setattr(self, k, file[self.tag][k][:])
                except: continue
            self.x = self.x1 
            self.y = self.x2
    def follow_video(self, background=None, window=100, **kwds):
        sim = self.species.parent
        if not background: background = sim.density
        di = sim.input.sp01.track_nstore
        db = sim.input.ndump
        center = (window//2, window//2)
        fig,ax,img = show(background[0], **kwds)
        line, = ax.plot(*center, ls='None', marker='.', ms=10, color='r')
        for i in range(0, sim.input.niter, di):
            pind = i // di 
            img.set

class dHybridRspecies(Species):
    def __init__(self, species_number: int, parent):
        self.n = self.number = self.sp_num = species_number
        self.nstr = str(self.n).zfill(2)
        self.parent = parent 
        self.input = getattr(parent.input, f"sp{str(self.n).zfill(2)}")
        self.loaded = False
        if self.input.track_dump:
            self.path = parent.path+f"/Output/Tracks/Sp{self.nstr}/track_Sp{self.nstr}.h5"
            self.load()
    
    def load(self):
        with h5File(self.path) as file:
            self.tags = np.array(list(file.keys()))
            self.loaded = True 

    def sample(self, N, load=False): return np.array([dHybridRparticle(t, self) for t in np.random.choice(self.tags, size=N, load=load)])

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
        self.outputDir = Folder(self.path+"/Output")
        if not self.outputDir.exists and verbose:
            if yesno("There is no output, would you like to run this simulation?\n"): 
                self.run()
        elif self.outputDir.exists: 
            self.parse_output()
            self.ncores: int = int(np.prod(self.input.node_number))
            self.ncores_charged: int = self.ncores + self.ncores % 128
            if self.outFile.exists:
                self.runtime: float = self.outFile.runtime #run time as calculated from out file, in hours
                self.corehours: float = self.runtime * np.prod(self.ncores)
                self.corehours_charged: float = self.runtime * self.ncores_charged
        self.restartDir = Folder(self.path+"/Restart")
    def __repr__(self) -> str: return self.name
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
        self.outFile = dHybridRout(self.path+"/out")
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
        if self.input.sp01.track_dump:
            self.sp01 = dHybridRspecies(1, self)

class TurbSim(dHybridR):
    def __init__(
            self, 
            path: str,
            caching: bool = False,
            verbose: bool = False,
            template: Folder = dHybridRtemplate,
            compressed: bool = False
        ) -> None:
        dHybridR.__init__(self, path, caching=caching, verbose=verbose, template=template, compressed=compressed)
        self.config = dHybridRconfig(self, mode='turb')
        
        self.initializer = TurbInit(self)