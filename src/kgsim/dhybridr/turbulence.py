# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.simulation import SimulationGroup
from kgsim.dhybridr.initializer import dHybridRinitializer, dHybridRconfig, dHybridRSnapshot
from kgsim.dhybridr.dhybridr import dHybridR

from kbasic import dHybridRtemplate, Folder, texfraction, where_between, progress_bar

import numpy as np 
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
line = lambda x, m, b: m*x+b
def field_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray: return np.sum(A * B, axis=0)
def efficiency(E, f, high_energy_threshold):
    total_energy = sum(E[:-1]*f[:-1]*np.diff(np.log(E)))
    non_thermal_energy = sum(E[high_energy_threshold:]*f[high_energy_threshold:]*np.diff(np.log(E))[high_energy_threshold-1:])
    return non_thermal_energy / total_energy
def non_thermal_slope(E, f, mach):
    fitting_zone = where_between(E, 5*mach**2, 10*mach**2)
    E = E[fitting_zone]
    f = f[fitting_zone]
    [slope, b], pcov = curve_fit(line, np.log10(E), np.log10(f/E))
    return slope, b
def S(p: int, u: np.ndarray, l: int, sample='all', verbose=True):
    """
    Compute structure function of power p, of field u, at lag l
    S_p(u, l) = <(u(x) - u(x + l))^p>
    p: int - order of structure function
    u: Array - field to take structure function of
    l: int - lag, i.e. how many pixels away should the function look
    """
    (Nx, Ny) = u.shape 
    assert l < min(u.shape), f"l ({l}) is larger than smallest dimension ({min(u.shape)})"
    grid = np.mgrid[-Nx//2:Nx//2, -Ny//2:Ny//2]
    grid = np.hypot(grid[0], grid[1])
    grid = np.where(grid//1 == l, True, False)
    in_annulus = lambda i, j: np.roll(grid, (i - Nx//2, j - Ny//2), axis=(0,1))
    # for some reason this needs to be transposed
    return np.array([
        [
            np.mean(
                (u[i, j] - u[in_annulus(i, j)]) ** p
            ) for i in np.arange(Nx)
        ] for j in progress_bar(np.arange(Ny))
    ]).T

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
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
                    l = self.input.niter if not self.simulation.output.exists else len(self.simulation.B)*self.input.ndump
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
                    l = self.input.niter if not self.simulation.output.exists else len(self.simulation.B)*self.input.ndump
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
        phases: np.ndarray = np.exp(2j * np.pi * np.random.random(field.shape))  # randomized complex phases
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
class TurbGroup(SimulationGroup):
    def __init__(self, path, sort='mach', verbose=True, **sim_kwds):
        SimulationGroup.__init__(self, path, simtype=TurbSim, **sim_kwds)
        try: 
            self.sort_by(sort)
        except KeyError: 
            if verbose: print(f"could not sort by {sort}, using default order...")
    def colorer(self, cmap=cm.plasma): 
        machs = [x.mach for x in self.simulations.values()]
        return [cmap(np.log2(m/min(machs))/np.log2((max(machs)+.1)/min(machs))) for m in machs]
    def labeler(self): return [
        r"$\mathcal{M} = $"+f"{int(x.mach) if x.mach.is_integer() else texfraction(x.mach)}" for x in self.simulations.values()]