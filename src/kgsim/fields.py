# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kplot import show, show_video
from kbasic.bar import verbose_bar
from kbasic.parsing import Folder, File
from kbasic.typing import Number, Iterable
from glob import glob
import numpy as np
from h5py import File as h5File
from functools import cached_property
from os.path import isdir, isfile
from matplotlib.pyplot import gca
from matplotlib.axes import Axes
from matplotlib.cm import plasma as default_cmap


# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def calc_psi(Bx, By, dx, dy):
    psi = np.zeros(Bx.shape)
    psi[1:,0] = np.cumsum(Bx[1:,0])*dy
    psi[:,1:] = (psi[:,0] - np.cumsum(By[:,1:], axis=1).T*dx).T
    return psi

def ddx(F,dx): return np.gradient(F, dx)[-1]
def ddy(F,dy): return np.gradient(F, dy)[-2] 
def ddz(F,dz): return np.gradient(F, dz)[-3]


def intdx(F,dx): return np.cumsum(F, axis=1) * dx
def intdy(F,dy): return np.cumsum(F, axis=0) * dy
def Az(Bx, By, dx=1, dy=1):
    Az = np.zeros(Bx.shape)
    Az[1:] = intdy(Bx[1:], dy)
    Az[:,1:] = (Az[:,0]-intdx(By[:,1:], dx).T).T
    return Az

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class ScalarField:
    """
    a special class of arrays used to efficiently interact with the fields output by simulations
    ________
    ~Inputs~
    * source - str | array-like
        the source of the scalar field. Can be a file, a folder full of files, or an array like object.
    ___________
    ~Atributes~
    * single - bool
        whether or this is a single field as opposed to a collection of fields
    * shape - tuple[int]
        the shape of output arrays a la numpy arrays
    * ndims - int 
        the number of dimensions in the output arrays

    ===FILE MODE=== 
    * path - str
        path containing field files 
    * file_names - list[str]
        the files containing fields
    * caching - bool
        whether or not to store file outputs for later use, more memory intensive but fewer file accesses
    * cache - dict
        a dictionary to store file outputs for later use
    * reader - function
        the function used to read files

    ===ARRAY MODE===
    * array - numpy.ndarray
        the array representing the field
    """
    def __init__(
        self, 
        source: str|Folder|File|list|np.ndarray, 
        parent = None,
        caching: bool = False,
        verbose: bool = False,
        name: str = None, 
        latex: str = None
    ) -> None:
        self.name: str = name 
        self.latex: str = latex
        self.verbose: bool = verbose
        self.parent = parent
        # if parent:
        #     self.di = tuple(np.array(parent.input.boxsize, dtype=float) / np.array(parent.input.ncells, dtype=int))
        #setup cache
        self.caching: bool = caching
        self.cache: dict = {}
        #find the correct constructor
        match source:
            #if its a folder
            case Folder():
                self.single = False
                example_file = File(source.children[0])
                if example_file.extension=="h5": self._from_folder_of_h5(source.path)
            case str() if isdir(source): 
                self.single = False
                files = glob(source+"/*")
                extension = files[0].split(".")[-1]
                if extension=="h5": self._from_folder_of_h5(source)
            #otherwise its a file
            case File():
                self.single = True
                if source.extension=="h5": self._from_h5(source.path)
            case str() if isfile(source): 
                self.single = True
                extension = source.split(".")[-1] 
                if extension=="h5": self._from_h5(source)
                else: self._from_csv(source)
            #or if its already been read
            case np.ndarray(): 
                self.single = True
                self._from_numpy(source)
            case list(): 
                self.single = True
                self._from_numpy(np.array(source))
    def __len__(self) -> int: return 1 if self.single else len(self.file_names)
    def __iter__(self):
        assert not self.single, "Cannot iterate through single scalar field"
        self.index = 0
        return self
    def __next__(self):
        if self.index < len(self):
            i = self.index
            self.index += 1
            return self[i]
        else: raise StopIteration
    def __getitem__(self, item: int|slice|tuple|list) -> np.ndarray:
        if self.single: return self.array[item]
        match item:
            case int(): 
                return self.cache[item] if self.caching and item in self.cache.keys() else self.reader(self.file_names[item], item)
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([
                    self.cache[i] if self.caching and i in self.cache.keys() else self.reader(self.file_names[i], i) for i in item_iters
                ])
            case tuple()|list(): return np.array([
                self.cache[i] if self.caching and i in self.cache.keys() else self.reader(self.file_names[i], i) for i in item
            ])

    def _from_folder_of_h5(self, path:str) -> None: 
        self.path = path.path if isinstance(path, Folder) else path
        self.file_names: list = sorted(glob(path + "/*.h5"))
        self.reader: function = self._read_h5_file
        self.shape = self[0].shape
        self.ndims = len(self.shape)
        self.size = np.prod(self.shape) * len(self)
    def _from_h5(self, file:str) -> None:
        self.file = file.path if isinstance(file,File) else file
        self.array = self._read_h5_file(file, 0)
        self.shape = self.array.shape 
        self.ndims = len(self.shape)
        self.size = np.prod(self.shape) * len(self)
    def _read_h5_file(self, file:str, item) -> np.ndarray:
        with h5File(file, 'r') as f:
            output = np.array(f["DATA"][:])
            #GODDMANIT I HATE THAT IT DOES Y,X and not X,Y
            if self.caching: self.cache[item] = output
            return output
    
    def _from_csv(self) -> None:
        self.single = True
    def _read_csv_file(self) -> None: pass
    
    def _from_numpy(self, array:np.ndarray) -> None:
        self.single = True
        self.array = array
        self.shape = array.shape
        self.ndims = len(self.shape)

    def show(self, item:int, **kwargs) -> None: 
        if hasattr(self.parent, 'dx'):
            x_ticks = np.arange(0, self.parent.input.boxsize[0], self.parent.dx)
            y_ticks = np.arange(0, self.parent.input.boxsize[1], self.parent.dy)
        show(self[item], x=x_ticks, y=y_ticks, **kwargs)
    
    def movie(self, norm='none', cmap=default_cmap, alter_func=None,**kwrg) -> None:
        @show_video(name=self.name, latex=self.latex, norm=norm, cmap=cmap)
        def reveal_thyself(s,alter_func=alter_func, **kwargs): 
            return np.array([self[i] for i in range(len(self))]) if alter_func is None else np.array([alter_func(self[i]) for i in range(len(self))])
        reveal_thyself(self if self.parent is None else self.parent, alter_func=alter_func,**kwrg)

class VectorField:
    def __init__(
            self, 
            *components, 
            caching: bool =False, 
            verbose: bool =False,
            name: str = None, 
            latex: str = None,
            parent = None,
            parallel: str = 'z'
        ) -> None:
        latex = "".join([c for c in latex if c not in r"$\{}"])
        self.name = name
        self.latex = latex 
        self.parent = parent
        # if parent: 
        #     self.dx, self.dy = parent.dx, parent.dy
        self.verbose = verbose 
        self.caching = caching
        child_kwargs = {'parent':parent, 'verbose':verbose, 'caching':caching}
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
    def __abs__(self) -> np.ndarray:
        homo = all([len(self.x[i])==len(self.x[0]) for i in range(len(self))])
        return np.array([
            np.sqrt(sum([
                c[i]**2 for c in self.components
            ])) for i in verbose_bar(range(len(self)), self.verbose, desc="taking magnitude...")
        ], dtype=float if homo else object)
    def __getitem__(self, item: int|slice) -> np.ndarray:
        match type(item):
            case int(): return np.array([c[item] for c in self.components])
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([
                    [
                        c[i] for c in self.components
                    ] for i in item_iters
                ])
    
    def dot(self, other) -> np.ndarray: 
        if isinstance(other, VectorField):
            assert self.ndims==3, "only 3D vector fields can be dotted at this time"
            assert other.ndims==3, "Can only dot into a 3D vector field at this time"
            if self.verbose: print("constructing A...")
            A = np.array([
                [
                    [
                        [self.x[k][i,j], self.y[k][i,j], self.z[k][i,j]]
                    for i in range(self.shape[0])]
                for j in range(self.shape[1])] 
            for k in verbose_bar(range(len(self)), self.verbose, desc="constructing A...")])
            B = np.array([
                [
                    [
                        [other.x[k][i,j], other.y[k][i,j], other.z[k][i,j]]
                    for i in range(other.shape[0])]
                for j in range(other.shape[1])] 
            for k in verbose_bar(range(len(other)), self.verbose, desc="constructing B...")])
            return np.sum(A * B, axis=2)
    def cross(self, other, k:int) -> np.ndarray:
        if type(other)==VectorField: 
            assert self.ndims==3, "only 3D vector fields can be crossed at this time"
            assert other.ndims==3, "Can only cross into a 3D vector field at this time"
            return np.array([
                self.y[k]*other.z[k] - self.z[k]*other.y[k], 
                self.z[k]*other.x[k] - self.x[k]*other.z[k],
                self.x[k]*other.y[k] - self.y[k]*other.x[k]
            ])
    @cached_property
    def potential(self):
        match len(self.x.shape):
            case 2: return np.array([Az(self.x[i], self.y[i], dx=self.dx, dy=self.dy) for i in range(len(self))])
    def curlz(self, item: int|slice, order: int = 2):
        match type(item):
            case int(): return curlz(self.x[item], self.y[item], order=order)
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([curlz(self.x[i], self.y[i], order=order) for i in item_iters])
    def div(self, item: int|slice, order: int = 2):
        if not item: return np.array([divergence(self.x[i], self.y[i], dx=self.dx, dy=self.dy, order=order) for i in range(len(self))])
        match type(item):
            case int(): return divergence(self.x[item], self.y[item], dx=self.dx, dy=self.dy, order=order)
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return np.array([divergence(self.x[i], self.y[i], dx=self.dx, dy=self.dy, order=order) for i in item_iters])
                

    def calc_Jz(self, item=None, verbose=True):
        if not item: self.Jz = np.array([np.nanstd(j) for j in verbose_bar(self.curlz(slice(0,len(self))), verbose)])
        elif type(item) in [int, slice]: self.Jz = np.nanstd(self.curlz(item), axis=(1,2))
        else: raise TypeError(f"calc_Jz only takes ints, slices, or None for item, not {type(item)}-type objects")

    def calc_perp(self, item=None) -> np.ndarray: 
        if not item:
            self.perp = np.array([
                np.hypot(self.perpendicular[0][j], self.perpendicular[1][j]) 
                for j in verbose_bar(range(len(self)), self.verbose, desc="perpendicularizing")
            ])
        elif type(item)==int: 
            self.perp = np.hypot(self.perpendicular[0][item], self.perpendicular[1][item])
        elif type(item)==slice: 
            item_iters = [
                i for i in range(
                    item.start if item.start else 0,
                    item.stop if item.stop else len(self),
                    item.step if item.step else 1
                )
            ]
            self.perp = np.array([np.hypot(self.perpendicular[0][j], self.perpendicular[1][j]) for j in item_iters])
        else: raise TypeError(f"calc_perp only takes ints, slices, or None for item, not {type(item)}-type objects")
    @cached_property
    def psi(self) -> np.ndarray: return np.array([
        calc_psi(Bx, By, self.dx, self.dy) for Bx, By in zip(self.x, self.y)
    ])       
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
                def reveal_thyself(s, **kwargs): return np.array([self.parallel[i] for i in range(len(self))])
        reveal_thyself(self if self.parent is None else self.parent, **kwrg)
    def quiver(
        self,
        ind: int,
        ax: Axes = None,
        #x/y axes
        density: float = 10,
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        transpose: bool = False,
        #everything else goes into matplotlib command
        **kwargs
    ):
        if not ax: ax = gca()
        match x, y:
            case None, None:
                dx = int(self.x[0].shape[1] // density)
                x = np.arange(0, self.x[0].shape[1], dx)
                dy = int(self.y[0].shape[0] // density)
                y = np.arange(0, self.x[0].shape[0], dy)
        # plot data
        print(dx, dy)
        if not transpose:
            ax.quiver(x, y, self.y[ind][::dy,::dx].T, self.x[ind][::dy,::dx].T, **kwargs)
        else: 
            ax.quiver(y, x, self.x[ind][::dy,::dx], self.y[ind][::dy,::dx], **kwargs)
class NablaOperator:
    __ins = None
    def __new__(cls):
        if cls.__ins is None:
            cls.__ins = super().__new__(cls)
        return cls.__ins
    def parse_grid_di(self, Field, di=None) -> tuple:
        match di:
            case Iterable(): return tuple(di)
            case _ if type(di) in Number.types: return (di, di, di) 
            case None:
                match Field:
                    case np.ndarray(): pass 
                    case ScalarField(parent=None): return (1, 1, 1)
                    case ScalarField(ndims=3): return (Field.parent.dz, Field.parent.dy, Field.parent.dx)
                    case ScalarField(ndims=2): return (Field.parent.dy, Field.parent.dx)
                    case VectorField(): return parse_grid_di(Field.x, di=di)
    def __mul__(self, other, di=None): # divergence
        match other:
            case VectorField():
                d = self.parse_grid_di(other.x)
                dz,dy,dx = d if len(d)==3 else tuple([np.inf, *d])
                dFxdx = ddx(other.x[:], dx)
                dFydy = ddy(other.y[:], dy)
                dFzdz = ddz(other.z[:], dz) if other.ndims==3 else 0
                return dFxdx + dFydy + dFzdz
    def x(self, other, di=1): # curl
        match other:
            case VectorField(): 
                d = self.parse_grid_di(other.x)
                dz,dy,dx = d if len(d)==3 else tuple([np.inf, *d])
                dt = 1 if not other.parent else other.parent.dt
                dFzdy = ddy(other.z[:], dy)
                dFydz = ddz(other.y[:], dz) if other.ndims==3 else 0 
                dFxdz = ddz(other.x[:], dz) if other.ndims==3 else 0 
                dFzdx = ddx(other.z[:], dx) 
                dFydx = ddx(other.y[:], dx)
                dFxdy = ddy(other.x[:], dy)
                return VectorField(
                    dFzdy - dFydz,
                    dFxdz - dFzdx,
                    dFydx - dFxdy,
                    caching=other.caching, verbose=other.verbose, parent=other.parent, 
                    name=f"Curl of {other.name}", latex=fr"$\Nabla \times {other.latex}"
                )
            case np.ndarray():
                d = self.parse_grid_di(other, di)
                dz,dy,dx = d if len(d)==3 else tuple([np.inf, *d])
                dFzdy = ddy(other[-3], dy) if len(d)==3 else 0
                dFydz = ddz(other[-2], dz) if len(d)==3 else 0 
                dFxdz = ddz(other[-1], dz) if len(d)==3 else 0 
                dFzdx = ddx(other[-3], dx) if len(d)==3 else 0
                dFydx = ddx(other[-2], dx)
                dFxdy = ddy(other[-1], dy)
                return np.array([
                    dFydx - dFxdy,
                    dFxdz - dFzdx,
                    dFzdy - dFydz
                ]) if len(d)==3 else dFydx - dFxdy
        
    def __call__(self, *args, **kwds): # gradient
        match args:
            case (ScalarField(),): 
                F: ScalarField = args[0]
                arr: np.ndarray = F[:]
                di: tuple[float] = F.di if 'di' not in kwds.keys() else kwds['di']
                gradarr: np.ndarray = np.gradient(arr, di)
                # assume that the 0th axis is time, as will be the case with all simulation outputs, but not necessarily numpy arrays
                # assume that the last axis is x
                return VectorField(
                    *gradarr[:0:-1],
                    caching=F.caching, verbose=F.verbose, name="grad-"+F.name, latex=fr"$\Nabla {F.latex[1:]}", parent=F.parent
                )
            case _: print("bad arguments : "+str(args)+str(kwds))

Nabla = NablaOperator()
Del = NablaOperator()
Grad = NablaOperator()