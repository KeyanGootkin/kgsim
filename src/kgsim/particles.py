"""particle support"""
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Imports                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from typing import Optional

from kgsim.fields.scalar import ScalarField
from kgsim.fields.vector import VectorField

from kbasic.typing import Number, Array
from kbasic.strings import purple
from kbasic.vectors import Vector
from kplot.axes import subplots
from kplot.cmaps import lch_cmap, auto_norm, Cmap, Norm, default_cmap, parse_color, colorbar
from kplot.plot import periodic_lines, line2segments
from kplot.image import show, move_image
from kplot.movie import func_video
from numpy import linspace, arange, array, ndarray, asarray, pad, zeros
from numpy.typing import ArrayLike
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                   Functions                                    <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def particle_interp(x: float, y: float, dx, dy, field):
    """blame colby if this fucks up >:)"""
    # position in cells
    xc, yc = x*dx, y*dy
    # grid points
    i = int(xc//1)
    j = int(yc//1)
    # weight factors
    wx = xc-i
    wy = yc-j
    # weighted terms
    fbl = field[i, j] * (1-wx) * (1-wy)
    ftl = field[i, j+1] * (1-wx) * wy
    ftr = field[i+1, j+1] * wx * wy
    fbr = field[i+1, j] * wx * (1-wy)
    return fbl+ftl+ftr+fbr
def lorentz_boost(u, v, c=100):
    """
    to confirm im not dumb http://hyperphysics.phy-astr.gsu.edu/hbase/Relativ/veltran.html
    u is the velocity measured in the first frame
    v is the velocity of the second frame in the first frame
    this works for kbasic.Vector's as well as scalars as well as arrays of vectors
    """
    return (u - v) / (1-abs(u)*abs(v)*c**-2)
def ExB_boost(u, E, B, c=100):
    """lorentz boost in the ExB drift frame"""
    v = E.cross(B) / abs(B)**2
    return lorentz_boost(u, v, c=c)


# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Classes                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class Species:
    def __init__(
            self,
            name: str,
            parent: Optional = None,
            m: float = 1.,
            q: float = 1.,
        ) -> None:
        """docstring"""
        self.name = name
        self.parent = parent
        self.m = m
        self.q = q
        self.mtq = m/q # mass to charge ratio
        self.qtm = q/m # charge to mass ratio
    def __repr__(self) -> str: return self.name
class Particle:
    def __init__(
            self,
            species: Species = Species("default"),
            tag: str = "0",
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
        # align the vectors
        match len(x), len(y), len(z):
            case nx, ny, nz if nx==ny and ny==nz:
                self.x, self.y, self.z = array(x), array(y), array(z)
            case nx, ny, 0 if nx==ny:
                self.x, self.y, self.z = array(x), array(y), zeros(nx)
            case nx, 0, 0:
                self.x, self.y, self.z = array(x), zeros(nx), zeros(nx)
            case _:
                raise ValueError("the sizes of x, y, z must be the same")
        self.position = Vector(self.x, self.y, self.z)
        match len(vx), len(vy), len(vz):
            case nx, ny, nz if nx==ny and ny==nz:
                self.vx, self.vy, self.vz = array(vx), array(vy), array(vz)
            case nx, ny, 0 if nx==ny:
                self.vx, self.vy, self.vz = array(vx), array(vy), zeros(nx)
            case nx, 0, 0:
                self.vx, self.vy, self.vz = array(vx), zeros(nx), zeros(nx)
            case _:
                raise ValueError("the sizes of x, y, z must be the same")
        self.velocity = Vector(self.vx, self.vy, self.vz)
        self.v = self.velocity
    def __len__(self) -> int: return len(self.position)
    def __repr__(self) -> str: return f"{self.species.name}: {self.tag}"
    def trail_segments(self, i: int, n_trail: int = 100, alpha='fade') -> tuple[list]:
        """docstring"""
        assert self.parent
        if i==0: return [], []
        n_trail = min(n_trail, i)
        i_start = i - n_trail
        trail = self.position[i_start: i]
        domain = ((0, self.parent.size[0]), (0, self.parent.size[1]))
        match alpha:
            case 'fade': a = linspace(0, 1, n_trail)
            case float(x): a = zeros(n_trail)+x
            case _: raise TypeError(f"alpha is {alpha} but must be float or 'fade'")
        xs, ys, alphas = periodic_lines(
            trail.x, trail.y, domain,
            alpha=a
        )
        segments = [s for x, y in zip(xs, ys) for s in line2segments(x, y)]
        segment_alphas = [ai for a in alphas for ai in a]
        return segments, segment_alphas
    def plot_trail(
        self,
        start: int = 0,
        end: Optional[int] = None,
        ax: Optional[Axes] = None,
        figsize: tuple[Number] = (5,5),
        alpha: float = 1.,
        color: tuple| str | Array = 'black',
        cmap: Cmap = default_cmap,
        norm: Norm | str = 'linear',
        cbar: dict = {
            'location':'right', "size":"7%", "pad":0.05, 'ticks':None, 'units': ''
        },
        save: bool | str = False
        ) -> tuple[Figure, Axes, LineCollection]:
        """docstring"""
        end = len(self) if not end else end
        if not ax: _,ax=subplots(figsize=figsize)
        fig = ax.get_figure()
        segments, segment_alphas = self.trail_segments(end, n_trail=end-start, alpha=alpha)
        norm = auto_norm(norm, color)
        color = parse_color(color, norm=norm, cmap=cmap)
        lc = LineCollection(segments, color=color)
        lc.set_alpha(segment_alphas)
        ax.add_collection(lc)
        ax.autoscale()
        c = cbar if cbar else {}
        colorbar(norm=norm, cmap=cmap, **c)
        if save:
            fname = save if isinstance(save, str) else 'default'
            fig.savefig(fname)
        return fig, ax, lc
    def distance_from(self, target_coords: tuple | Vector):
        """docstring"""
        tc = Vector(target_coords)
        assert tc.ndims==self.position.ndims
        return abs(tc - self.position)
    def interpolate_onto(self, field) -> Array:
        """docstring"""
        dx = self.parent.dx
        dy = self.parent.dy
        match field:
            case ScalarField():
                field = [fi for i in range(self.species.pipsi) for fi in field]
                return array([
                    particle_interp(xi, yi, dx, dy, fi) for xi, yi, fi in zip(self.x, self.y, field)
                ])
            case VectorField():
                comps = []
                for fieldi in field.components:
                    comp_i = [fi for i in range(self.species.pipsi) for fi in fieldi]
                    ci = array([
                        particle_interp(xi, yi, dx, dy, fi)\
                        for xi, yi, fi in zip(self.x, self.y, comp_i)
                    ])
                    comps.append(ci)
                print(comps)
                return Vector(*comps)
        raise TypeError("giveme a better field pls")

class Population:
    def __init__(self, particles) -> None:
        self.particles = particles
        self.tags = [p.tag for p in self.particles]
    def __new__(cls, particles):
        match particles:
            case Population(): return particles
            case _:
                return super().__new__(cls)
    def __getitem__(self, item) -> Particle | list[Particle]:
        match item:
            case int()|slice():
                return self.particles[item]
            case str():
                return self.particles[self.tags==item]
            case Array():
                return [self[i] for i in item]

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                   Functions                                    <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
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
    species = pop[0].species
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
        segs, alphas = p.trail_segments(i=10, n_trail=trail)
        trail_segments += list(segs)
        trail_alphas += list(alphas)
    lc = LineCollection(trail_segments, capstyle='butt', color=color)
    lc.set_alpha(trail_alphas)
    #plot trails
    def update(i: int):
        bi = int(i)
        pi = int(i * species.pipsi)
        img.set_array(background[bi])
        trail_segments = []
        trail_alphas = []
        for part, marker in zip(particles, particle_markers):
            marker.set_offsets([part.x[pi], part.y[pi]])
            segs, alphas = part.trail_segments(i=pi, n_trail=trail)
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
