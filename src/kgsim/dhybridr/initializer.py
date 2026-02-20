# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.fields import ScalarField, VectorField

from kbasic.parsing import File, Folder
from scipy.io import FortranFile
import numpy as np
from numpy import pi
from collections.abc import Iterable

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def parse_config_value(val:str):
    """take a string representing the value of a configuration parameter and figure out what python type it should be. 

    Args:
        val (str): The string representing the value after the = for each line of the configuration file

    Returns:
        str | int | float | np.ndarray: the python object corresponding to the configuration value
    """
    if "," in val: return np.array([float(x) for x in val.split(',')])
    elif "." in val: return float(val)
    elif val[0] in "1234567890": return int(val)
    else: return val

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class dHybridRconfig(File):
    def __init__(self, parent, mode: str|None = None) -> None:
        """Container for the information needed to initialize a dHybridR simulation

        Args:
            parent (dHybridR): the simulation this config file belongs to
            mode (str | None, optional): Represents what kind of simulation this is. Defaults to None.
        """
        self.params = []
        self.parent = parent 
        path = f"{self.parent.path}/config"
        File.__init__(self, path) #initialize as a File object
        #if the file already exists, read its contents
        if self.exists: self.read()
        #else user must supply the information
        else: self.set_interactive()

    def __setattr__(self, name, value) -> None:
        if name!="params": self.params.append(name)
        return super().__setattr__(name, value)

    def read(self) -> None:
        """method to read in the parameters in the config file and assign those to this object."""
        self.params = []
        with open(self.path, 'r') as file:
            self.lines = file.readlines()
            #for each line in the file
            for line in self.lines:
                line = line.strip()
                #skip comments and blank lines
                if any([
                    len(line)==0,
                    line.startswith("#"),
                    line.startswith("!")
                ]): continue
                # split the variable name and value and asign to this object
                name, value = (x.strip() for x in line.split("="))
                self.params.append(name)
                setattr(self, name, parse_config_value(value))
    
    def write(self) -> None:
        """write each of the parameters in the param attribute to the config file. WARNING: Overwrites config file.
        """
        #construct one line for each parameter in params
        self.lines = [f"{n}={getattr(self,n)}" for n in self.params]
        #write the file
        with open(self.path, 'w') as file: file.write("\n".join(self.lines))

    def set_interactive(self) -> None:
        print("this simulation hasn't been configured yet.")
        print("send an empty parameter to complete and write this object to file.")
        self.mode = input("What mode? ").lower()
        while len(new_param:=input("param: ").strip())>0:
            self.__setattr__(new_param, parse_config_value(input("value: ")))
            print("\n")
        self.write()

class dHybridRSnapshot:
    def __init__(
        self, 
        parent,
        i: int,
        caching: bool = False,
        verbose: bool = False
    ):
        i = i if i<len(parent.B) else len(parent.B)-1
        kwargs = {'caching':caching, 'verbose':verbose, 'parent':parent}
        print(parent, i, parent.B.x[i])
        self.B = VectorField(parent.B.x[i], parent.B.y[i], parent.B.z[i], name=parent.B.name, latex=parent.B.latex, **kwargs)
        self.u = VectorField(parent.u.x[i], parent.u.y[i], parent.u.z[i], name=parent.u.name, latex=parent.u.latex, **kwargs)
        self.E = VectorField(parent.E.x[i], parent.E.y[i], parent.E.z[i], name=parent.E.name, latex=parent.E.latex, **kwargs)
        self.density = ScalarField(parent.density[i], name=parent.density.name, latex=parent.density.latex, **kwargs)
        self.energy_grid = parent.energy_grid[i]
        self.energy_pdf = parent.energy_pdf[i]
        self.dlne = parent.dlne[i]
        self.T = sum(self.energy_grid*self.energy_pdf*self.dlne)
        self.tau = parent.tau[i]
        self.time = parent.time[i]

class dHybridRinitializer:
    def __init__(
        self,
        simulation
    ):
        self.simulation = simulation
        self.input = self.simulation.input
        self.dims = len(self.input.ncells)
        self.L = self.input.boxsize
        #parse grid size and shape from input
        self.build()

    def build(self):
        match self.dims:
            case 1:
                self.dx: float = self.L[0] / self.input.ncells[0]
                self.Nx: int = int(self.input.ncells[0])
                self.shape: tuple = self.Nx,
            case 2:
                [self.dy, self.dx] = np.array(self.L) / np.array(self.input.ncells)
                [self.Ny, self.Nx] = self.input.ncells
                self.shape = (self.Ny, self.Nx)
            case 3:
                [self.dy, self.dx, self.dz] = np.array(self.L) / np.array(self.input.ncells)
                [self.Ny, self.Nx, self.Nz] = self.input.ncells
                self.shape = (self.Ny, self.Nx, self.Nz)

    def build_B_field(self): 
        self.B = np.array([np.zeros(self.input.ncells) for i in range(2)])
    def build_u_field(self):
        self.u = np.array([np.zeros(self.input.ncells) for i in range(2)])
    def save_init_field(self, field: np.ndarray, path: str): 
        FortranFile(path, 'w').write_record(field)
    def prepare_simulation(self):
        self.build_B_field()
        self.save_init_field(self.B.T, self.simulation.path+"/input/Bfld_init.unf")
        self.build_u_field()
        self.save_init_field(self.u.T, self.simulation.path+"/input/vfld_init.unf")

class FlareWaveInit(dHybridRinitializer):
    def __init__(
            self,
            input_file,
            B0 = 1,
            Bg = 2,
            w0 = 2,
            psi0 = 0.5,
            vth = 0.1
    ):
        dHybridRinitializer.__init__(self, input_file)
        self.B0 = B0
        self.Bg = Bg 
        self.w0 = w0 
        self.psi0 = psi0

    def build_B_field(self, unknown_variable=69.12):
        x = np.arange(0, self.Nx) * self.dx
        y = np.arange(0, self.Ny) * self.dy
        Bx = np.array([
            self.B0 * (np.tanh((y - 0.25*self.L[1])/self.w0) - np.tanh((y - 0.75*self.L[1])/self.w0) - 1)
        for i in range(len(x))]).T
        By = np.array([
            (unknown_variable / self.L[0]) * np.cos(2*np.pi*x / self.L[0]) * np.sin(2*np.pi*x / self.L[0])**10
        for i in range(len(y))])
        Bz = np.sqrt(self.B0**2 + self.Bg**2 - Bx**2)
        self.B = np.array([Bx.T, By.T, Bz.T], dtype=np.float32)   