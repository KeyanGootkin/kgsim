# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.fields import ScalarField, VectorField
from kgsim.dhybridr.io import dHybridRinput

from kbasic.parsing import File, Folder
from scipy.io import FortranFile
import numpy as np
from numpy import pi
from collections.abc import Iterable

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def field_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray: return np.sum(A * B, axis=0)
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

class TurbInit(dHybridRinitializer):
    def __init__(
        self,
        simulation
    ):
        self.simulation = simulation
        self.dims = len(self.simulation.input.boxsize)
        self.config = dHybridRconfig(simulation)
        self.mach = self.config.mach
        self.simulation.mach = self.mach 
        self.dB = self.config.dB
        self.simulation.dB = self.dB
        self.amplitude: tuple = (self.dB, self.mach)
        #this gives us L, N's, shape, and d's, as well as the basic code to produce B and u fields and save those as d files
        dHybridRinitializer.__init__(self, simulation)
        
        self.configure()

    def configure(self):
        #config works different for simulations of different dimensions
        match self.dims:
            case 1: print("not implemented, low-key not sure you can do this in dHybridR????")
            case 2:
                #set the initial k space annuli for producing turbulence
                if "kinit" not in self.config.params: self.kinit = (1, np.pi), (1, np.pi) #Default value if config file has no kinit
                else: #if the config file has a kinit value use that
                    if len(self.config.kinit)==2: self.kinit = self.config.kinit, self.config.kinit 
                    elif len(self.config.kinit)==4: self.kinit = self.config.kinit[:2], self.config.kinit[2:]
                    else: raise ValueError(f"config file's kinit value is invalid\nkinit={self.config.kinit}")
                    #set the simulations kinit value for future use
                self.simulation.kinit = self.kinit

                #set k range 
                self.kmin = 2 * np.pi / max(self.L)
                self.kmax = 2 * np.pi / min([self.dx, self.dy]) #is this even used for anything? I don't think so?

                #set k vectors and compute their magnitude
                self.k = np.mgrid[
                    -self.Ny // 2: self.Ny // 2,
                    -self.Nx // 2: self.Nx // 2
                ][::-1] * self.kmin
                self.kmag = np.hypot(*self.k)
                self.kmag[self.kmag==0] = np.nan

                if not self.simulation.compressed:
                    l = self.input.niter if not self.simulation.outputDir.exists else len(self.simulation.B)*self.input.ndump
                    self.simulation.time = np.arange(0, l, self.input.ndump) * self.input.dt
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                elif self.simulation.compressed:
                    self.simulation.time = np.array([int(x[-11:-3]) for x in self.simulation.density.file_names]) * self.simulation.input.dt
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                # if "peak_jz" in self.config.params: 
                #     self.simulation.peak_jz_ind = int(self.config.peak_jz)
                #     self.simulation.initial = dHybridRSnapshot(self.simulation,0)
                #     self.simulation.peak = dHybridRSnapshot(self.simulation, self.simulation.peak_jz_ind)
                #     self.simulation.snapshots = [
                #         self.simulation.initial, self.simulation.peak
                #     ]+[
                #         dHybridRSnapshot(self.simulation, np.argmin(abs(self.simulation.tau - n))) for n in range(1, int(self.simulation.tau[-1]//1))
                #     ]
            case 3:
                #set the initial k space annuli for producing turbulence
                if "kinit" not in self.config.params: self.kinit = (1, 2*np.pi), (1, 2*np.pi), (1, 2*np.pi) #Default value if config file has no kinit
                else: #if the config file has a kinit value use that
                    if len(self.config.kinit)==2: self.kinit = self.config.kinit, self.config.kinit, self.config.kinit
                    # if there are two sets then assume it goes perp, par -> kinit_perp, kinit_perp, kinit_par
                    elif len(self.config.kinit)==4: self.kinit = self.config.kinit[:2], self.config.kinit[:2], self.config.kinit[2:]
                    elif len(self.config.kinit)==6: self.kinit = self.config.kinit[:2], self.config.kinit[2:4], self.config.kinit[4:]
                    else: raise ValueError(f"config file's kinit value is invalid\nkinit={self.config.kinit}")
                #set the simulations kinit value for future use
                self.simulation.kinit = self.kinit
                #set k range 
                self.kmin = 2 * np.pi / max(self.L)
                self.kmax = 2 * np.pi / min([self.dx, self.dy, self.dz]) #is this even used for anything? I don't think so?

                #set k vectors and compute their magnitude
                self.k = np.mgrid[
                    -self.Ny // 2: self.Ny // 2,
                    -self.Nx // 2: self.Nx // 2,
                    -self.Nz // 2: self.Nz // 2
                ][::-1] * self.kmin
                self.kmag = np.sqrt(self.k[0]**2 + self.k[1]**2 + self.k[2]**2)
                self.kmag[self.kmag==0] = np.nan
                #set times
                if not self.simulation.compressed:
                    l = self.input.niter if not self.simulation.outputDir.exists else len(self.simulation.B)*self.input.ndump
                    self.simulation.time = np.arange(0, l, self.input.ndump) * self.input.dt
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                elif self.simulation.compressed:
                    self.simulation.time = np.array([int(x[-11:-3]) for x in self.simulation.density.file_names]) * self.simulation.input.dt
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                if "peak_jz" in self.config.params: 
                    self.simulation.peak_jz_ind = int(self.config.peak_jz)
                    self.simulation.initial = dHybridRSnapshot(self.simulation,0)
                    self.simulation.peak = dHybridRSnapshot(self.simulation, self.simulation.peak_jz_ind)
                    self.simulation.snapshots = [
                    self.simulation.initial, self.simulation.peak
                    ]+[
                    dHybridRSnapshot(self.simulation, np.argmin(abs(self.simulation.tau - n))) for n in range(1, int(self.simulation.tau[-1]//1))
                    ]

    def fluctuate3D(self, field, amp, no_div=True):
        init_mask = np.array([
            np.where((self.kinit[0][0] * self.kmin < self.kmag)&(self.kmag < self.kinit[0][1]*self.kmin),True,False),
            np.where((self.kinit[1][0] * self.kmin < self.kmag)&(self.kmag < self.kinit[1][1]*self.kmin),True,False),
            np.where((self.kinit[2][0] * self.kmin < self.kmag)&(self.kmag < self.kinit[2][1]*self.kmin),True,False)
        ])
        M = np.sum(init_mask)
        phases = np.exp(2j * np.pi * np.random.random(field.shape))

        FT = np.zeros(field.shape, dtype=complex)
        FT[0][init_mask[0]] = amp[0] * np.pi / 2
        FT[1][init_mask[1]] = amp[1] * np.pi
        FT[2][init_mask[2]] = amp[2] * np.pi / 2
        FT *= phases
        # subtract off the parallel x/y components
        if no_div: FT -= field_dot(FT, self.k / self.kmag) * self.k / self.kmag
        FT[np.isnan(FT)] = 0
        # apply the condition to make this real
        _fx = np.roll(FT[1, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[1, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fx[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[1, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = np.conj(_fx[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[1, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = np.conj(_fx[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[1, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = np.conj(_fx[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])
        _fy = np.roll(FT[0, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[0, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fy[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[0, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fy[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[0, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = np.conj(_fy[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[0, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = np.conj(_fy[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[0, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = np.conj(_fy[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])
        _fz = np.roll(FT[2, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[2, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fz[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[2, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = np.conj(_fz[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[2, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = np.conj(_fz[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[2, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = np.conj(_fz[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[2, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = np.conj(_fz[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])

        self.FT = FT
        # take the inverse fourier transform
        y: np.ndarray = np.real(
            np.fft.ifftn(
                np.fft.ifftshift(
                    FT
                )
            )
        ) / M * self.Nx * self.Ny * self.Nz
        rms = np.sqrt(np.nanmean(y[0]**2 + y[1]**2 + y[2]**2))
        factor = np.array(amp / rms)
        y[0] = factor[0] * y[0]
        y[1] = factor[1] * y[1]
        y[2] = factor[2] * y[2]
        return np.float32(y)

    def fluctuate2D(self, field, amp, no_div=True):
        """
        Given the initialization create a 2d array the same shape as the simulation which will smoothly fluctuate
        over length scales kinit
        :return y: np.ndarray[np.float32]: random fluctuations set by parameters passed to __init__
        """
        init_mask: np.ndarray = np.where((self.kinit[0][0] * self.kmin < self.kmag) & (self.kmag < self.kinit[0][1] * self.kmin))
        M: int = len(init_mask[0])  # number of cells with an amplitude
        phases: np.ndarray = np.exp(2j * pi * np.random.random(field.shape))  # randomized complex phases
        phases[2] *= 0  # don't wiggle the z component
        # Setting the fourier transform
        FT: np.ndarray = np.zeros(field.shape, dtype=complex)  # same shape as field
        FT[0][init_mask] = amp * np.pi / 2 # set x and y amplitudes
        FT[1][init_mask] = amp * np.pi
        FT *= phases  # apply phases
        # subtract off the parallel x/y components
        if no_div:
            FT[:2] -= field_dot(FT[:2], self.k / self.kmag) * self.k / self.kmag
        FT[np.isnan(FT)] = 0
        # apply the condition to make this real
        _fx = np.roll(FT[1, ::-1, ::-1], 1, axis=(0, 1))
        FT[1, :self.Ny // 2] = np.conj(_fx[:self.Ny // 2])

        _fy = np.roll(FT[0, ::-1, ::-1], 1, axis=(0, 1))
        FT[0, :self.Ny // 2] = np.conj(_fy[:self.Ny // 2])

        # I think we have to fix the zero line
        FT[1, self.Ny // 2, 1:self.Nx // 2] = FT[1, self.Ny // 2, self.Nx // 2 + 1:][::-1]
        FT[1, self.Ny // 2, 1:self.Nx // 2] = np.conj(FT[1, self.Ny // 2, self.Nx // 2 + 1:][::-1])
        FT[1, self.Ny // 2, :] = 0.j
        FT[1, :, self.Nx // 2] = 0.j

        FT[0, self.Ny // 2, 1:self.Nx // 2] = FT[0, self.Ny // 2, self.Nx // 2 + 1:][::-1]
        FT[0, self.Ny // 2, 1:self.Nx // 2] = np.conj(FT[0, self.Ny // 2, self.Nx // 2 + 1:][::-1])
        
        self.FT = FT

        # take the inverse fourier transform
        y: np.ndarray = np.array([*np.real(
            np.fft.ifft2(
                np.fft.ifftshift(
                    FT[:2]
                )
            )
        ), np.zeros(FT[0].shape)]) / M * self.Nx * self.Ny

        rms = np.sqrt(np.nanmean(y[0]**2 + y[1]**2))
        y *= (amp / rms)
        return np.float32(y)
    
    def fluctuate(self, field, amp, no_div=True):
        match self.dims:
            case 2: return self.fluctuate2D(field, amp, no_div=no_div)
            case 3: return self.fluctuate3D(field, amp, no_div=no_div)

    def construct_field(self, x, y, z, amp, no_div=True):
        """
        Constructs a 3 x N x N array representing a constant x, y, and z component with additional fluctuations
        :param x:
        :param y:
        :param z:
        :param no_div: whether or not to ensure that the divergence of the field is 0 when applying fluctuations
        :return field:
        """
        base_field = np.array([
            np.zeros(self.shape) + y,
            np.zeros(self.shape) + x,
            np.zeros(self.shape) + z
        ], dtype=np.float32)

        base_rms: float = np.sqrt(np.mean(base_field ** 2))
        fluctuations: np.ndarray = self.fluctuate(base_field, amp, no_div=no_div)
        field: np.ndarray = base_field + fluctuations
        alt_rms: float = np.sqrt(np.mean(base_field ** 2))
        if not any([base_rms == 0, alt_rms == 0]):
            field *= base_rms / alt_rms
        return field
    
    def build_B_field(self): self.B = self.construct_field(0, 0, 1, self.amplitude[0])
    def build_u_field(self): self.u = self.construct_field(0, 0, 0, self.amplitude[1])

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