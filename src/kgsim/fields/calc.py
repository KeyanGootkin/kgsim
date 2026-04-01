# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.fields.scalar import ScalarField
from kgsim.fields.vector import VectorField
from kbasic.typing import ArrayLike, Array, Number 
from numpy import gradient, cumsum, ndarray, inf
from numpy.typing import NDArray
from typing import Optional

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def ddx(F: ArrayLike, dx: Number) -> NDArray: return gradient(F, dx)[-1]
def ddy(F: ArrayLike, dy: Number) -> NDArray: return gradient(F, dy)[-2] 
def ddz(F: ArrayLike, dz: Number) -> NDArray: return gradient(F, dz)[-3]
def intdx(F: ArrayLike, dx: Number) -> NDArray: return cumsum(F, axis=1) * dx
def intdy(F: ArrayLike, dy: Number) -> NDArray: return cumsum(F, axis=0) * dy

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class NablaOperator:
    __ins = None
    def __new__(cls):
        if cls.__ins is None:
            cls.__ins = super().__new__(cls)
        return cls.__ins
    def parse_grid_di(self, Field, di=None) -> tuple:
        match di:
            case _ if type(di) in Array.types: return tuple(di)
            case _ if type(di) in Number.types: return (di, di, di) 
            case None:
                match Field:
                    case ndarray(): pass 
                    case ScalarField(parent=None): return (1, 1, 1)
                    case ScalarField(ndims=3): return (Field.parent.dz, Field.parent.dy, Field.parent.dx)
                    case ScalarField(ndims=2): return (Field.parent.dy, Field.parent.dx)
                    case VectorField(): return self.parse_grid_di(Field.x, di=di)
    def __mul__(
            self, 
            other: VectorField, 
            di: Optional[Number|ArrayLike] = None
            ) -> NDArray: # divergence
        match other:
            case VectorField():
                d = self.parse_grid_di(other.x)
                dz,dy,dx = d if len(d)==3 else tuple([inf, *d])
                dFxdx = ddx(other.x[:], dx)
                dFydy = ddy(other.y[:], dy)
                dFzdz = ddz(other.z[:], dz) if other.ndims==3 else 0
                return dFxdx + dFydy + dFzdz
    def x(
            self, 
            other: VectorField | NDArray
            ) -> VectorField | NDArray: # curl
        match other:
            case VectorField(): 
                d = self.parse_grid_di(other.x)
                dz,dy,dx = d if len(d)==3 else tuple([inf, *d])
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
            case ndarray():
                d = self.parse_grid_di(other, di)
                dz,dy,dx = d if len(d)==3 else tuple([inf, *d])
                dFzdy = ddy(other[-3], dy) if len(d)==3 else 0
                dFydz = ddz(other[-2], dz) if len(d)==3 else 0 
                dFxdz = ddz(other[-1], dz) if len(d)==3 else 0 
                dFzdx = ddx(other[-3], dx) if len(d)==3 else 0
                dFydx = ddx(other[-2], dx)
                dFxdy = ddy(other[-1], dy)
                return array([
                    dFydx - dFxdy,
                    dFxdz - dFzdx,
                    dFzdy - dFydz
                ]) if len(d)==3 else dFydx - dFxdy
    def __call__(self, *args, **kwds) -> VectorField: # gradient
        match args:
            case (ScalarField(),): 
                F: ScalarField = args[0]
                arr: ndarray = F[:]
                di: tuple[float] = F.di if 'di' not in kwds.keys() else kwds['di']
                gradarr: ndarray = gradient(arr, di)
                # assume that the 0th axis is time, as will be the case with all simulation outputs, but not necessarily numpy arrays
                # assume that the last axis is x
                return VectorField(
                    *gradarr[:0:-1],
                    caching=F.caching, verbose=F.verbose, name="grad-"+F.name, latex=fr"$\Nabla {F.latex[1:]}", parent=F.parent
                )
            case _: print("bad arguments : "+str(args)+str(kwds))

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
Nabla = NablaOperator()
Del = NablaOperator()
Grad = NablaOperator()
