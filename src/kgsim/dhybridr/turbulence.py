# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.simulation import SimulationGroup
from kgsim.templates import dHybridRtemplate
from kgsim.dhybridr.initializer import dHybridRinitializer, dHybridRconfig, dHybridRSnapshot
from kgsim.dhybridr.dhybridr import dHybridR

from kbasic import Folder, texfraction, where_between, progress_bar

from collections.abc import Iterable
from numpy import ndarray, diff, log10, mgrid, hypot, where, roll, array, mean, \
                  arange, pi, sqrt, nan, argmin, exp, zeros, isnan, conj, real, \
                  nanmean, float32, log2
from numpy import sum as nsum
from numpy.random import random
from numpy.fft import ifftn, ifft2, ifftshift
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
line = lambda x, m, b: m*x+b
def field_dot(A: ndarray, B: ndarray) -> ndarray: return nsum(A * B, axis=0)
def efficiency(E, f, high_energy_threshold):
    total_energy = sum(E[:-1]*f[:-1]*diff(log10(E)))
    non_thermal_energy = sum(E[high_energy_threshold:]*f[high_energy_threshold:]*diff(log10(E))[high_energy_threshold-1:])
    return non_thermal_energy / total_energy
def non_thermal_slope(E, f, mach):
    fitting_zone = where_between(E, 5*mach**2, 10*mach**2)
    E = E[fitting_zone]
    f = f[fitting_zone]
    [slope, b], pcov = curve_fit(line, log10(E), log10(f/E))
    return slope, b
def S(p: int, u: ndarray, l: int, sample='all', verbose=True):
    """
    Compute structure function of power p, of field u, at lag l
    S_p(u, l) = <(u(x) - u(x + l))^p>
    p: int - order of structure function
    u: Array - field to take structure function of
    l: int - lag, i.e. how many pixels away should the function look
    """
    (Nx, Ny) = u.shape 
    assert l < min(u.shape), f"l ({l}) is larger than smallest dimension ({min(u.shape)})"
    grid = mgrid[-Nx//2:Nx//2, -Ny//2:Ny//2]
    grid = hypot(grid[0], grid[1])
    grid = where(grid//1 == l, True, False)
    in_annulus = lambda i, j: roll(grid, (i - Nx//2, j - Ny//2), axis=(0,1))
    # for some reason this needs to be transposed
    return array([
        [
            mean(
                (u[i, j] - u[in_annulus(i, j)]) ** p
            ) for i in arange(Nx)
        ] for j in progress_bar(arange(Ny))
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
                if "kinit" not in self.config.params: self.kinit = (1, pi), (1, pi) #Default value if config file has no kinit
                else: #if the config file has a kinit value use that
                    if len(self.config.kinit)==2: self.kinit = self.config.kinit, self.config.kinit 
                    elif len(self.config.kinit)==4: self.kinit = self.config.kinit[:2], self.config.kinit[2:]
                    else: raise ValueError(f"config file's kinit value is invalid\nkinit={self.config.kinit}")
                    #set the simulations kinit value for future use
                self.simulation.kinit = self.kinit

                #set k range 
                self.kmin = 2 * pi / max(self.L)
                self.kmax = 2 * pi / min([self.dx, self.dy]) #is this even used for anything? I don't think so?

                #set k vectors and compute their magnitude
                self.k = mgrid[
                    -self.Ny // 2: self.Ny // 2,
                    -self.Nx // 2: self.Nx // 2
                ][::-1] * self.kmin
                self.kmag = hypot(*self.k)
                self.kmag[self.kmag==0] = nan

                if not self.simulation.compressed:
                    l = self.input.niter if not self.simulation.output.exists else len(self.simulation.B)*self.input.ndump
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                elif self.simulation.compressed:
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                # if "peak_jz" in self.config.params: 
                #     self.simulation.peak_jz_ind = int(self.config.peak_jz)
                #     self.simulation.initial = dHybridRSnapshot(self.simulation,0)
                #     self.simulation.peak = dHybridRSnapshot(self.simulation, self.simulation.peak_jz_ind)
                #     self.simulation.snapshots = [
                #         self.simulation.initial, self.simulation.peak
                #     ]+[
                #         dHybridRSnapshot(self.simulation, argmin(abs(self.simulation.tau - n))) for n in range(1, int(self.simulation.tau[-1]//1))
                #     ]
            case 3:
                #set the initial k space annuli for producing turbulence
                if "kinit" not in self.config.params: self.kinit = (1, 2*pi), (1, 2*pi), (1, 2*pi) #Default value if config file has no kinit
                else: #if the config file has a kinit value use that
                    if len(self.config.kinit)==2: self.kinit = self.config.kinit, self.config.kinit, self.config.kinit
                    # if there are two sets then assume it goes perp, par -> kinit_perp, kinit_perp, kinit_par
                    elif len(self.config.kinit)==4: self.kinit = self.config.kinit[:2], self.config.kinit[:2], self.config.kinit[2:]
                    elif len(self.config.kinit)==6: self.kinit = self.config.kinit[:2], self.config.kinit[2:4], self.config.kinit[4:]
                    else: raise ValueError(f"config file's kinit value is invalid\nkinit={self.config.kinit}")
                #set the simulations kinit value for future use
                self.simulation.kinit = self.kinit
                #set k range 
                self.kmin = 2 * pi / max(self.L)
                self.kmax = 2 * pi / min([self.dx, self.dy, self.dz]) #is this even used for anything? I don't think so?

                #set k vectors and compute their magnitude
                self.k = mgrid[
                    -self.Ny // 2: self.Ny // 2,
                    -self.Nx // 2: self.Nx // 2,
                    -self.Nz // 2: self.Nz // 2
                ][::-1] * self.kmin
                self.kmag = sqrt(self.k[0]**2 + self.k[1]**2 + self.k[2]**2)
                self.kmag[self.kmag==0] = nan
                #set times
                if not self.simulation.compressed:
                    l = self.input.niter if not self.simulation.output.exists else len(self.simulation.B)*self.input.ndump
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                elif self.simulation.compressed:
                    self.simulation.tau = self.simulation.time * max(self.mach if isinstance(self.mach, Iterable) else [self.mach]) / (max(self.input.boxsize))
                if "peak_jz" in self.config.params: 
                    self.simulation.peak_jz_ind = int(self.config.peak_jz)
                    self.simulation.initial = dHybridRSnapshot(self.simulation,0)
                    self.simulation.peak = dHybridRSnapshot(self.simulation, self.simulation.peak_jz_ind)
                    self.simulation.snapshots = [
                    self.simulation.initial, self.simulation.peak
                    ]+[
                    dHybridRSnapshot(self.simulation, argmin(abs(self.simulation.tau - n))) for n in range(1, int(self.simulation.tau[-1]//1))
                    ]
    def fluctuate3D(self, field, amp, no_div=True):
        init_mask = array([
            where((self.kinit[0][0] * self.kmin < self.kmag)&(self.kmag < self.kinit[0][1]*self.kmin),True,False),
            where((self.kinit[1][0] * self.kmin < self.kmag)&(self.kmag < self.kinit[1][1]*self.kmin),True,False),
            where((self.kinit[2][0] * self.kmin < self.kmag)&(self.kmag < self.kinit[2][1]*self.kmin),True,False)
        ])
        M = sum(init_mask)
        phases = exp(2j * pi * random.random(field.shape))

        FT = zeros(field.shape, dtype=complex)
        FT[0][init_mask[0]] = amp[0] * pi / 2
        FT[1][init_mask[1]] = amp[1] * pi
        FT[2][init_mask[2]] = amp[2] * pi / 2
        FT *= phases
        # subtract off the parallel x/y components
        if no_div: FT -= field_dot(FT, self.k / self.kmag) * self.k / self.kmag
        FT[isnan(FT)] = 0
        # apply the condition to make this real
        _fx = roll(FT[1, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[1, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = conj(_fx[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[1, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = conj(_fx[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[1, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = conj(_fx[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[1, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = conj(_fx[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])
        _fy = roll(FT[0, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[0, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = conj(_fy[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[0, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = conj(_fy[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[0, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = conj(_fy[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[0, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = conj(_fy[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[0, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = conj(_fy[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])
        _fz = roll(FT[2, ::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
        FT[2, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = conj(_fz[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[2, :self.Ny // 2, :self.Nx // 2, :self.Nz // 2] = conj(_fz[:self.Ny // 2, :self.Nx // 2, :self.Nz // 2])
        FT[2, self.Ny // 2:, :self.Nx // 2, :self.Nz // 2] = conj(_fz[self.Ny // 2:, :self.Nx // 2, :self.Nz // 2])
        FT[2, :self.Ny // 2, :self.Nx // 2, self.Nz // 2:] = conj(_fz[:self.Ny // 2, :self.Nx // 2, self.Nz // 2:])
        FT[2, self.Ny // 2:, :self.Nx // 2, self.Nz // 2:] = conj(_fz[self.Ny // 2:, :self.Nx // 2, self.Nz // 2:])

        self.FT = FT
        # take the inverse fourier transform
        y: ndarray = real(
            ifftn(
                ifftshift(
                    FT
                )
            )
        ) / M * self.Nx * self.Ny * self.Nz
        rms = sqrt(nanmean(y[0]**2 + y[1]**2 + y[2]**2))
        factor = array(amp / rms)
        y[0] = factor[0] * y[0]
        y[1] = factor[1] * y[1]
        y[2] = factor[2] * y[2]
        return float32(y)
    def fluctuate2D(self, field, amp, no_div=True):
        """
        Given the initialization create a 2d array the same shape as the simulation which will smoothly fluctuate
        over length scales kinit
        :return y: ndarray[float32]: random fluctuations set by parameters passed to __init__
        """
        init_mask: ndarray = where((self.kinit[0][0] * self.kmin < self.kmag) & (self.kmag < self.kinit[0][1] * self.kmin))
        M: int = len(init_mask[0])  # number of cells with an amplitude
        phases: ndarray = exp(2j * pi * random.random(field.shape))  # randomized complex phases
        phases[2] *= 0  # don't wiggle the z component
        # Setting the fourier transform
        FT: ndarray = zeros(field.shape, dtype=complex)  # same shape as field
        FT[0][init_mask] = amp * pi / 2 # set x and y amplitudes
        FT[1][init_mask] = amp * pi
        FT *= phases  # apply phases
        # subtract off the parallel x/y components
        if no_div:
            FT[:2] -= field_dot(FT[:2], self.k / self.kmag) * self.k / self.kmag
        FT[isnan(FT)] = 0
        # apply the condition to make this real
        _fx = roll(FT[1, ::-1, ::-1], 1, axis=(0, 1))
        FT[1, :self.Ny // 2] = conj(_fx[:self.Ny // 2])

        _fy = roll(FT[0, ::-1, ::-1], 1, axis=(0, 1))
        FT[0, :self.Ny // 2] = conj(_fy[:self.Ny // 2])

        # I think we have to fix the zero line
        FT[1, self.Ny // 2, 1:self.Nx // 2] = FT[1, self.Ny // 2, self.Nx // 2 + 1:][::-1]
        FT[1, self.Ny // 2, 1:self.Nx // 2] = conj(FT[1, self.Ny // 2, self.Nx // 2 + 1:][::-1])
        FT[1, self.Ny // 2, :] = 0.j
        FT[1, :, self.Nx // 2] = 0.j

        FT[0, self.Ny // 2, 1:self.Nx // 2] = FT[0, self.Ny // 2, self.Nx // 2 + 1:][::-1]
        FT[0, self.Ny // 2, 1:self.Nx // 2] = conj(FT[0, self.Ny // 2, self.Nx // 2 + 1:][::-1])
        
        self.FT = FT

        # take the inverse fourier transform
        y: ndarray = array([*real(
            ifft2(
                ifftshift(
                    FT[:2]
                )
            )
        ), zeros(FT[0].shape)]) / M * self.Nx * self.Ny

        rms = sqrt(nanmean(y[0]**2 + y[1]**2))
        y *= (amp / rms)
        return float32(y)   
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
        base_field = array([
            zeros(self.shape) + y,
            zeros(self.shape) + x,
            zeros(self.shape) + z
        ], dtype=float32)

        base_rms: float = sqrt(mean(base_field ** 2))
        fluctuations: ndarray = self.fluctuate(base_field, amp, no_div=no_div)
        field: ndarray = base_field + fluctuations
        alt_rms: float = sqrt(mean(base_field ** 2))
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
        return [cmap(log2(m/min(machs))/log2((max(machs)+.1)/min(machs))) for m in machs]
    def labeler(self): return [
        r"$\mathcal{M} = $"+f"{int(x.mach) if x.mach.is_integer() else texfraction(x.mach)}" for x in self.simulations.values()]