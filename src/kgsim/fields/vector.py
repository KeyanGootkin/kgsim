# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.fields.scalar import ScalarField
from kbasic.parsing import Folder
from kbasic.typing import ArrayLike
from kbasic.bar import verbose_bar
from kplot import default_cmap, show_video
from functools import cached_property
from typing import Optional, Self
from numpy.typing import NDArray
from numpy import array, sqrt, zeros, nanstd, arange, cumsum, hypot
from matplotlib.pyplot import gca 
from matplotlib.axes import Axes

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def Az(Bx, By, dx=1, dy=1):
    Az = zeros(Bx.shape)
    Az[1:] = cumsum(Bx[1:]*dy, axis=0)
    Az[:,1:] = (Az[:,0]-cumsum(By[:,1:]*dx, axis=1).T).T
    return Az

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class VectorField:
    def __init__(
            self, 
            *components, 
            stats: Optional[Folder|str] = None,
            caching: bool = False, 
            verbose: bool = False,
            name: Optional[str] = None, 
            latex: Optional[str] = None,
            parent = None,
            parallel: Optional[str] = 'z'
        ) -> None:
        latex = "".join([c for c in latex if c not in r"$\{}"])
        self.stats = Folder(stats)
        self.name = name
        self.latex = latex 
        self.parent = parent
        if parent: 
            self.dx, self.dy = parent.dx, parent.dy
        self.verbose = verbose 
        self.caching = caching
        child_kwargs = {'parent':parent, 'verbose':verbose, 'caching':caching, 'stats': stats}
        if len(components)==1 and type(path:=components[0])==str:
            components = (
                ScalarField(path+"/x", name=name+"_x_component", latex=f"${latex}_x$", **child_kwargs), 
                ScalarField(path+"/y", name=name+"_y_component", latex=f"${latex}_y$", **child_kwargs), 
                ScalarField(path+"/z", name=name+"_z_component", latex=f"${latex}_z$", **child_kwargs)
            )
        self.ndims = len(components)
        assert 1<self.ndims<4, f"Only 2-3 components are supported. {len(components)} were given"
        component_names = "xyz"
        self.components = []
        for name,val in zip(component_names, components): 
            comp = ScalarField(val, caching=self.caching) if type(val)==str else val
            self.components.append(comp)
            setattr(self, name, comp)
        self.size = sum([c.size for c in self.components])
        self.set_parallel(parallel)
    def __len__(self) -> int: return min([len(self.x), len(self.y), len(self.z)])
    def __abs__(self) -> NDArray:
        homo = all([len(self.x[i])==len(self.x[0]) for i in range(len(self))])
        return array([
            sqrt(sum([
                c[i]**2 for c in self.components
            ])) for i in verbose_bar(range(len(self)), self.verbose, desc="taking magnitude...")
        ], dtype=float if homo else object)
    def __getitem__(self, item: int|slice) -> NDArray:
        match type(item):
            case int(): return array([c[item] for c in self.components])
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return array([
                    [
                        c[i] for c in self.components
                    ] for i in item_iters
                ])
    def dot(self, other: Self) -> NDArray: 
        if isinstance(other, VectorField):
            assert self.ndims==3, "only 3D vector fields can be dotted at this time"
            assert other.ndims==3, "Can only dot into a 3D vector field at this time"
            if self.verbose: print("constructing A...")
            A = array([
                [
                    [
                        [self.x[k][i,j], self.y[k][i,j], self.z[k][i,j]]
                    for i in range(self.shape[0])]
                for j in range(self.shape[1])] 
            for k in verbose_bar(range(len(self)), self.verbose, desc="constructing A...")])
            B = array([
                [
                    [
                        [other.x[k][i,j], other.y[k][i,j], other.z[k][i,j]]
                    for i in range(other.shape[0])]
                for j in range(other.shape[1])] 
            for k in verbose_bar(range(len(other)), self.verbose, desc="constructing B...")])
            return sum(A * B, axis=2)
    def cross(self, other: Self, k:int) -> NDArray:
        if type(other)==VectorField: 
            assert self.ndims==3, "only 3D vector fields can be crossed at this time"
            assert other.ndims==3, "Can only cross into a 3D vector field at this time"
            return array([
                self.y[k]*other.z[k] - self.z[k]*other.y[k], 
                self.z[k]*other.x[k] - self.x[k]*other.z[k],
                self.x[k]*other.y[k] - self.y[k]*other.x[k]
            ])
    @cached_property
    def potential(self) -> NDArray:
        match len(self.x.shape):
            case 2: 
                return array([Az(self.x[i], self.y[i], dx=self.dx, dy=self.dy) for i in range(len(self))])
    def calc_perp(self, item=None) -> NDArray: 
        if not item:
            self.perp = array([
                hypot(self.perpendicular[0][j], self.perpendicular[1][j]) 
                for j in verbose_bar(range(len(self)), self.verbose, desc="perpendicularizing")
            ])
        elif type(item)==int: 
            self.perp = hypot(self.perpendicular[0][item], self.perpendicular[1][item])
        elif type(item)==slice: 
            item_iters = [
                i for i in range(
                    item.start if item.start else 0,
                    item.stop if item.stop else len(self),
                    item.step if item.step else 1
                )
            ]
            self.perp = array([hypot(self.perpendicular[0][j], self.perpendicular[1][j]) for j in item_iters])
        else: raise TypeError(f"calc_perp only takes ints, slices, or None for item, not {type(item)}-type objects")
    def set_parallel(self, component:str) -> None:
        match component.lower():
            case 'x':
                self.parallel = self.x 
                self.perpendicular = self.y, self.z 
            case 'y':
                self.parallel = self.y 
                self.perpendicular = self.x, self.z 
            case 'z':
                self.parallel = None if not hasattr(self, 'z') else self.z
                self.perpendicular = self.x, self.y 
    def movie(self, mode='mag', norm='none', cmap=default_cmap, **kwrg) -> None:
        match mode.lower():
            case 'mag'|'magnitude'|'abs':
                @show_video(name=self.name+"_magnitude", latex=fr"$|{self.name}|$", norm=norm, cmap=cmap)
                def reveal_thyself(s, **kwargs): return abs(self)
            case 'perp'|'perpendicular':
                @show_video(name=self.name+"_perp", latex=fr"${self.name}_\perp$", norm=norm, cmap=cmap)
                def reveal_thyself(s, **kwargs): return self.perp                
            case 'par'|'parallel':
                @show_video(name=self.name+"_par", latex=fr"${self.name}_\parallel$", norm=norm, cmap=cmap)
                def reveal_thyself(s, **kwargs): return array([self.parallel[i] for i in range(len(self))])
        reveal_thyself(self if self.parent is None else self.parent, **kwrg)
    def quiver(
        self,
        ind: int,
        ax: Optional[Axes] = None,
        #x/y axes
        density: float = 10,
        x: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        transpose: bool = False,
        #everything else goes into matplotlib command
        **kwargs
    ) -> None:
        if not ax: ax = gca()
        match x, y:
            case None, None:
                dx = int(self.x[0].shape[1] // density)
                x = arange(0, self.x[0].shape[1], dx)
                dy = int(self.y[0].shape[0] // density)
                y = arange(0, self.x[0].shape[0], dy)
        # plot data
        print(dx, dy)
        if not transpose:
            ax.quiver(x, y, self.y[ind][::dy,::dx].T, self.x[ind][::dy,::dx].T, **kwargs)
        else: 
            ax.quiver(y, x, self.x[ind][::dy,::dx], self.y[ind][::dy,::dx], **kwargs)
