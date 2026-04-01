"""particle support"""
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Imports                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from typing import Optional

from kgsim.fields.scalar import ScalarField

# from kbasic.typing import Array, Number
from kbasic.strings import purple
from kplot.cmaps import lch_cmap, auto_norm, Cmap, Norm
from kplot.plot import periodic_lines, line2segments
from kplot.image import show, move_image
from kplot.movie import func_video
from numpy import linspace, arange, array, ndarray, asarray, pad
from numpy.typing import NDArray, ArrayLike
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Classes                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class Species:
    def __init__(
            self,
            name: str,
            m: float = 1.,
            q: float = 1.
        ) -> None:
        """docstring"""
        self.name = name
        self.m = m
        self.q = q
        self.mtq = m/q # mass to charge ratio
        self.qtm = q/m # charge to mass ratio
    def __repr__(self) -> str: return self.name
class Particle:
    def __init__(
            self,
            species: Species = Species("default"),
            tag: Optional[str] = "0",
            x: ArrayLike = array([]),
            y: ArrayLike = array([]),
            z: ArrayLike = array([]),
            vx: ArrayLike = array([]),
            vy: ArrayLike = array([]),
            vz: ArrayLike = array([]),
            t: ArrayLike = array([])
        ) -> None:
        """docstring"""
        self.species = species
        self.parent = species.parent
        self.tag = tag
        self.t = array(t)
        self.x, self.y, self.z = array(x), array(y), array(z)
        self.vx, self.vy, self.vz = array(vx), array(vy), array(vz)
        self.position: tuple = (self.x, self.y, self.z)
        self.velocity: tuple = (self.vx, self.vy, self.vz)

    def __repr__(self) -> str: return f"{self.species.name}: {self.tag}"
    def trail_segments(self, i: int, n_trail: int = 100) -> tuple[list]:
        i_start = i - n_trail if n_trail<i else 0
        if n_trail>i: ntrail = i
        coords = None
    def distance_from(self, target_coords: tuple, ind: Optional[int] = None):
        """docstring"""
        assert len(target_coords)==len(self.position)
class Population:
    def __init__(self, particles) -> None:
        self.particles = particles
    def __new__(cls, particles) -> None:
        match particles:
            case Population(): return particles
            case [Particle(), *_]: super().__new__(cls)
            case _: raise TypeError("Population's can only be instatiated with a list of particles")
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                   Functions                                    <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def particle_trail_segments(p: Particle, i: int, N: int) -> tuple[list]:
    """docstring"""
    sim = p.parent
    i_start = i - N if N<i else 0
    N_trail = N if N<i else i
    x, y = p.x[i_start: i], p.y[i_start: i]
    domain = ((0, sim.input.boxsize[0]), (0, sim.input.boxsize[1]))
    xs, ys, alphas = periodic_lines(x, y, domain, alpha=linspace(0,1,N_trail))
    segments = []
    segment_alphas = []
    for x, y, a in zip(xs, ys, alphas):
        seg = line2segments(x, y)
        segments += list(seg)
        segment_alphas += list(a)
    return segments, segment_alphas
def video_particle_over(
        vname: str,
        particles: list[Particle] | Population,
        background: ndarray|ScalarField,
        ax: Optional[Axes] = None,
        axis_off: bool = False,
        trail: int = 200, # length of the particle trails
        color: tuple | str = 'white',
        cmap: Cmap | str = lch_cmap(hue=(-100,100)),
        norm: Norm | str = 'linear',
        verbose: bool = True,
    ) -> None:
    """docstring"""
    pop = Population(particles)
    sim = pop[0].parent
    #setup background
    Lx, Ly = sim.input.boxsize
    x_grid = arange(0, Lx, sim.dx)
    y_grid = arange(0, Ly, sim.dy)
    norm = auto_norm(norm, background)
    #plot background
    fig,ax,img = show(
        background[0], 
        ax=ax, norm=norm, x=x_grid, y=y_grid, cmap=cmap, colorbar=False
        )
    if axis_off: ax.set_axis_off()
    #setup trails
    trail_segments = []
    trail_alphas = []
    #setup particle markers
    particle_markers = []
    for p in particles:
        #plot point
        particle_markers.append(ax.scatter(p.x[0], p.y[0], color=color, s=3))
        #add particle data to trails
        segs, alphas = particle_trail_segments(p, 10, trail)
        trail_segments += list(segs)
        trail_alphas += list(alphas)
    lc = LineCollection(trail_segments, capstyle='butt', color=color)
    lc.set_alpha(trail_alphas)
    #plot trails
    ax.add_collection(lc)
    def update(i: int):
        bi = int(i)
        pi = int(i * species.pipsi)
        img.set_array(background[bi])
        trail_segments = []
        trail_alphas = []
        for part, marker in zip(particles, particle_markers):
            marker.set_offsets([part.x[pi], part.y[pi]])
            segs, alphas = particle_trail_segments(part, pi, trail)
            trail_segments += list(segs)
            trail_alphas += list(alphas)
        if asarray(trail_segments).size == 0: return None
        lc.set_segments(trail_segments)
        lc.set_alpha(trail_alphas)
    func_video(
        vname, fig, update, len(background)-1,
        verbose=verbose, frames=f'~/.temp/{vname}'
    )
def follow_particle_video(
        vname: str, part: Particle, background: ndarray, 
        window=20, 
        norm='linear',
        ax=None,
        cmap=lch_cmap(luminosity=(.2,1),hue=(-100,100)),
        verbose: bool = True
    ) -> None:
    """docstring"""
    if verbose: print(purple("Loading..."))
    species = part.species
    sim = species.parent
    #setup background
    Lx, Ly = sim.input.boxsize
    x_grid = arange(-(window+1) * sim.dx, Lx + (window+1) * sim.dx, sim.dx)
    y_grid = arange(-(window+1) * sim.dy, Ly + (window+1) * sim.dy, sim.dy)
    if verbose: print(purple("Calculating norm..."))
    norm = auto_norm(norm, background)
    #plot background
    if verbose: print(purple("Initializing plot..."))
    fig,ax,img = show(
        pad(background[0], window+1, mode='wrap'), 
        ax=ax, 
        norm=norm, 
        x=x_grid, y=y_grid, 
        cmap=cmap
        )
    ax.scatter(0,0)
    ax.set_xlim(-window*sim.dx, window*sim.dx)
    ax.set_ylim(-window*sim.dy, window*sim.dy)
    def update(i: int):
        bi = int(i)
        pi = int(i * species.pipsi)
        img.set_array(pad(background[bi], window+1, mode='wrap'))
        move_image(img, (0.5-part.x[pi])*sim.dx, (0.5-part.y[pi])*sim.dy)
    if verbose: print(purple("Rendering..."))
    func_video(vname, fig, update, len(background)-1, verbose=verbose)
