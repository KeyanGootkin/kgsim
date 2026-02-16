# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
#pysim imports
from kgsim.parsing import Folder
from kgsim.fields import ScalarField, VectorField
from kgsim.simulation import GenericSimulation
from kgsim.dhybridr.io import InputParameter
#non-pysim imports
from glob import glob
import numpy as np
from h5py import File

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# keys are parameters and values are the type of file to read those from
tristan_param_file_convert = {
    'bx': 'flds.tot', 
    'by': 'flds.tot', 
    'bz': 'flds.tot', 
    'dens': 'flds.tot', 
    'densi': 'flds.tot', 
    'ex': 'flds.tot', 
    'ey': 'flds.tot', 
    'ez': 'flds.tot', 
    'jx': 'flds.tot', 
    'jy': 'flds.tot', 
    'jz': 'flds.tot', 
    'v3x': 'flds.tot', 
    'v3xi': 'flds.tot', 
    'v3y': 'flds.tot', 
    'v3yi': 'flds.tot', 
    'v3z': 'flds.tot', 
    'v3zi': 'flds.tot',
    'che': 'prtl.tot', 
    'chi': 'prtl.tot', 
    'gammae': 'prtl.tot', 
    'gammai': 'prtl.tot', 
    'inde': 'prtl.tot', 
    'indi': 'prtl.tot', 
    'proce': 'prtl.tot', 
    'proci': 'prtl.tot', 
    'time': 'prtl.tot', 
    'ue': 'prtl.tot', 
    'ui': 'prtl.tot', 
    've': 'prtl.tot', 
    'vi': 'prtl.tot', 
    'we': 'prtl.tot', 
    'wi': 'prtl.tot', 
    'xe': 'prtl.tot', 
    'xi': 'prtl.tot', 
    'ye': 'prtl.tot', 
    'yi': 'prtl.tot', 
    'ze': 'prtl.tot', 
    'zi': 'prtl.tot',
    'acool': 'param', 
    'bphi': 'param', 
    'btheta': 'param', 
    'c': 'param', 
    'c_omp': 'param', 
    'caseinit': 'param', 
    'cooling': 'param', 
    'delgam': 'param', 
    'dlapion': 'param', 
    'dlaplec': 'param', 
    'dummy': 'param', 
    'gamma0': 'param', 
    'interval': 'param', 
    'istep': 'param', 
    'istep1': 'param', 
    'me': 'param', 
    'mi': 'param', 
    'mx': 'param', 
    'mx0': 'param', 
    'my': 'param', 
    'my0': 'param', 
    'mz0': 'param', 
    'ntimes': 'param', 
    'pltstart': 'param', 
    'ppc0': 'param', 
    'qi': 'param', 
    'sigma': 'param', 
    'sizex': 'param', 
    'sizey': 'param', 
    'stride': 'param', 
    'testendion': 'param', 
    'testendlec': 'param', 
    'teststarti': 'param', 
    'teststartl': 'param', 
    'time': 'param', 
    'torqint': 'param', 
    'walloc': 'param', 
    'xinject2': 'param',
    'gamma': 'spectl', 
    'gmax': 'spectl', 
    'gmin': 'spectl', 
    'spece': 'spectl', 
    'specerest': 'spectl', 
    'specp': 'spectl', 
    'specprest': 'spectl', 
    'xsl': 'spectl'
}

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def tristan_loader(
        sim_path: str, inds: list[int], params: list[str], 
        zfill_level: int = 3, padding: int = 3
) -> dict:
    # Colby wrote this one I think?
    output = {}
    param_files = [tristan_param_file_convert[pi] for pi in params]
    # for each file we need to read
    for pf in np.unique(param_files):
        for i in inds:
            with File(f"{sim_path}/output/{pf}.{str(i+1).zfill(zfill_level)}") as file:
                # for each parameter requested
                for pi in params:
                    if pi not in file: continue # only take the params in the active file
                    P = file[pi][:].copy()
                    if P.ndim==3: P = P[0, padding : -padding + 1, padding : -padding + 1]
                    output[pi] = np.append(output[pi], P, axis=0) if pi in output else P
    return output

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class TristanInput:
    def __init__(
            self,
            path: str,
    ):
        """A class for reading/storing info from Tristan input files.

        Args:
            path (str): path to the input file.
        """
        self.path = path
        with open(self.path, 'r') as file:
            self.lines = file.readlines()
        for line in self.lines:
            line = line.strip()
            if line.startswith(("#", '<')) or len(line)==0: continue
            param, *valcom = line.split('=')
            valcom = "=".join(valcom)
            value, comment = valcom.split('#') if '#' in valcom else (valcom.strip(), None)
            param, value = param.strip(), value.strip()
            setattr(self, param, InputParameter(param, float(value), comment=comment))

class TristanScalarField(ScalarField):
    def __init__(
            self,
            param_key: str,
            parent = None,
            caching: bool = False,
            verbose: bool = False,
            name: str = None,
            latex: str = None,
            zfill_level: int = 3,
            padding: int = 3
    ):
        """A ScalarField object specialized for Tristan output files.

        Args:
            param_key (str): key used to store info in output files
            parent (Tristan, optional): The parent simultion from which to read outputs. Defaults to None.
            caching (bool, optional): Whether to store field values after reading. Defaults to False.
            verbose (bool, optional): Doesn't do much rn but might be good for debugging later. Defaults to False.
            name (str, optional): Name of field to be used in e.g. saving plots of. Defaults to None.
            latex (str, optional): Name of field to be used when using latex. Defaults to None.
            zfill_level (int, optional): Number of 0's to use when constructing file names. Defaults to 3.
            padding (int, optional): number of overflow cells surrounding simulation domain. Defaults to 3.
        """
        ScalarField.__init__(
            self,
            None,
            parent=parent,
            caching=caching,
            verbose=verbose,
            name=name,
            latex=latex
        )
        self.param_key = param_key
        self.zfill_level = zfill_level
        self.padding = padding
        self.single = False
        self.file_prefix = tristan_param_file_convert[self.param_key]
        self.file_names = glob(f"{self.parent.path}/output/{tristan_param_file_convert[self.param_key]}.*")

    def reader(self, fname: str, item: int):
        output = tristan_loader(self.parent.path, [item], [self.param_key], zfill_level=self.zfill_level, padding=self.padding)
        return output[self.param_key]

class TristanParticleQuantity(TristanScalarField):
    def __init__(            
            self,
            param_key: str,
            parent = None,
            caching: bool = False,
            verbose: bool = False,
            name: str = None,
            latex: str = None
    ):
        """Object for storing Tristan particle data

        Args:
            param_key (str): key used to access output data
            parent (Tristan, optional): Simulation from which output is drawn. Defaults to None.
            caching (bool, optional): Whether to store data after reading. Defaults to False.
            verbose (bool, optional): Doesn't do much for now. Defaults to False.
            name (str, optional): Name to use in e.g. filenames. Defaults to None.
            latex (str, optional): name to use in latex. Defaults to None.
        """
        TristanScalarField.__init__(
            self,
            param_key,
            parent = parent,
            caching = caching,
            verbose = verbose,
            name = name,
            latex = latex,
            padding = 0
        )

class TristanParticleSpecies:
    def __init__(
            self,
            name: str,
            parent = None,
            m: float = 1.,
            q: float = 1.,
            N: int = None,
            sig: str = ""
    ):
        """Object to store data for a particle species from a Tristan simulation.

        Args:
            name (str): Name of the particle species
            parent (Tristan, optional): Simulation from which to draw data. Defaults to None.
            m (float, optional): mass. Defaults to 1..
            q (float, optional): charge. Defaults to 1..
            N (int, optional): number of particles. Defaults to None.
            sig (str, optional): signature to add to the end of keys e.g. 'i' for ions. Defaults to "".
        """
        self.name = name 
        self.m = m 
        self.q = q 
        self.N = N
        self.sig = sig
        self.parent = parent
        if parent is not None:
            # load actual parameters
            kwargs = {'caching':parent.caching, 'verbose':parent.verbose, 'parent':parent}
            self.x = TristanParticleQuantity("x"+sig, name="x", latex="$x$", **kwargs)
            self.y = TristanParticleQuantity("y"+sig, name="y", latex="$y$", **kwargs)
            self.z = TristanParticleQuantity("z"+sig, name="z", latex="$z$", **kwargs)
            self.v = VectorField(
                TristanParticleQuantity("u"+sig, name="vx", latex="$v_x$", **kwargs),
                TristanParticleQuantity("v"+sig, name="vy", latex="$v_y$", **kwargs),
                TristanParticleQuantity("w"+sig, name="vz", latex="$v_z$", **kwargs),
                name="v", latex=r"$\vec{v}$"
            )
            self.gamma = TristanParticleQuantity("gamma"+sig, name="gamma", latex=rf"$\gamma_{sig}$", **kwargs)
            self.ch = TristanParticleQuantity("ch"+sig, **kwargs)
            self.Bx = TristanParticleQuantity("bx", name="Bx", latex="$B_x$", **kwargs)

            # self.density,_,_ = np.array([np.histogram2d(self.x[i], self.y[i], bins=[self.parent.Nx, self.parent.Ny], weights=self.ch) for i in range(len(self.x))])

class Tristan(GenericSimulation):
    def __init__(
            self,
            path: str,
            template: str|Folder = None,
            caching: bool = False,
            verbose: bool = False,
            compressed: bool = False
    ):
        """Object to interact with Tristan simulations.

        Args:
            path (str): path to simulation top-level
            template (str | Folder, optional): template from which to compare. Defaults to None.
            caching (bool, optional): whether to store data after reading it in. Defaults to False.
            verbose (bool, optional): Doesn't do much rn. Defaults to False.
            compressed (bool, optional): Whether this is a full simulation or just some timestamps. Defaults to False.
        """
        self.compressed = compressed
        GenericSimulation.__init__(self, path, template=template, caching=caching, verbose=verbose)
        self.parse_input()
        self.parse_output()

    def __len__(self): return len(glob(f"{self.path}/output/flds.tot.*"))
    def __repr__(self): return f"Tristan Simulation: {self.path}"

    def parse_input(self):
        """Read input file and construct relevant quantities
        """
        self.input = TristanInput(self.path+"/input")
        # time
        self.c = self.input.c.value
        self.steps = np.arange(0, self.input.last.value, self.input.interval.value)
        w_pe = self.steps[-1] * self.c / self.input.c_omp.value
        om_ce = w_pe * np.sqrt(self.input.sigma.value)
        mratio = self.input.mi.value / self.input.me.value
        om_ci = om_ce / mratio
        self.time = np.round(self.steps / om_ci, 0)
        # grids
        self.Nx = int(self.input.mx0.value)
        self.Ny = int(self.input.my0.value)
        self.Nz = int(self.input.mz0.value)
        self.shape = (self.Nx, self.Ny, self.Nz)
        d_xe = self.Nx * self.input.istep.value / self.input.c_omp.value
        d_xi = d_xe / self.input.mi.value 
        d_ye = self.Ny * self.input.istep.value / self.input.c_omp.value
        d_yi = d_ye / self.input.mi.value
        self.x = np.linspace(0, d_xe, self.Nx)
        self.y = np.linspace(0, d_ye, self.Ny)
        self.dx = np.diff(self.x)[0]
        self.dy = np.diff(self.y)[0]

    def parse_output(self):
        """Read output files and construct relevant data structures.
        """
        kwargs = {'caching':self.caching, 'verbose':self.verbose, 'parent':self}
        self.B = VectorField(
            TristanScalarField("bx", name='bx', latex='$B_x$', **kwargs),
            TristanScalarField("by", name='by', latex='$B_y$', **kwargs),
            TristanScalarField("bz", name='bz', latex='$B_z$', **kwargs),
            name='magnetic', latex=r"$\vec{B}$", **kwargs
        )
        self.E = VectorField(
            TristanScalarField("ex", name='ex', latex='$E_x$', **kwargs),
            TristanScalarField("ey", name='ey', latex='$E_y$', **kwargs),
            TristanScalarField("ez", name='ez', latex='$E_z$', **kwargs),
            name='electric', latex=r"$\vec{E}$", **kwargs
        )
        self.J = VectorField(
            TristanScalarField("jx", name='jx', latex='$J_x$', **kwargs),
            TristanScalarField("jy", name='jy', latex='$J_y$', **kwargs),
            TristanScalarField("jz", name='jz', latex='$J_z$', **kwargs),
            name='electric', latex=r"$\vec{J}$", **kwargs
        )
        self.u = VectorField(
            TristanScalarField("v3x", name='ux', latex='$u_x$', **kwargs),
            TristanScalarField("v3y", name='uy', latex='$u_y$', **kwargs),
            TristanScalarField("v3z", name='uz', latex='$u_z$', **kwargs),
            name='bulk-flow', latex=r"$\vec{u}$", **kwargs
        )
        self.density = TristanScalarField("dens", name='density', latex=r"$\rho$", **kwargs)
        self.ions = TristanParticleSpecies("proton", parent=self, m=self.input.mi.value, q=1, sig='i')
        self.electrons = TristanParticleSpecies("electron", parent=self, m=self.input.me.value, q=-1, sig='e')

    def load(self, params: list[str], item: int|slice = slice(None, None)):
        """loads data from params at item times.

        Args:
            params (list[str]): keys representing data to load
            item (int | slice, optional): which timestamps to load data for. Defaults to slice(None, None).

        Returns:
            data (dict): a dictionary with params as keys and values corresponding to the data at each time in item.
        """
        match item:
            case int(): 
                items = [item]
            case slice(): 
                items = [
                    x for x in range(
                        0 if item.start is None else item.start, 
                        len(self) if item.stop is None else item.stop,
                        1 if item.step is None else item.step
                        )
                    ]
        return tristan_loader(self.path, items, params)