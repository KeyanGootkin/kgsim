"""Objects to handle spacial field data"""
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Imports                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from typing import Optional, Self, Callable
from glob import glob
from functools import cached_property
from os.path import isdir, isfile
import logging

from kbasic.parsing import File, Folder, ensure_path, configure_log
from kbasic.typing import Array, Number
from kplot import show, default_cmap, show_video
from numpy.typing import NDArray, ArrayLike
from numpy import ndarray, array, prod, arange, nanmin, nanmax, nanmean, nanstd, \
                  nanmedian, inf, append, float32, loadtxt, savetxt
from h5py import File as h5File

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                                    Classes                                     <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class ScalarField:
    """
    a special class of arrays used to efficiently interact with the fields output by simulations
    ________
    ~Inputs~
    * source - str | array-like
        the source of the scalar field. Can be a file, a folder full of files, 
        or an array like object.
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
        whether or not to store file outputs for later use, more memory intensive but 
        fewer file accesses
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
        source: str|Folder|File|Array,
        name: Optional[str] = None,
        latex: Optional[str] = None,
        parent: Optional = None,
        stats: Optional[Folder | str] = None,
        caching: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        self.name: str = name
        self.latex: str = latex
        self.parent = parent
        self.log = self.parent.log.getChild(name) if parent else \
            configure_log(
                f"{__name__}.ScalarField.{name}",
                console_level=logging.DEBUG if debug else logging.INFO if verbose else None
                )
        self.log.info("initializing...")
        self.stats = stats
        #setup cache
        self.caching: bool = caching
        self.cache: dict = {}
        #find the correct constructor
        match source:
            #if its a folder
            case Folder() | str() if isdir(source):
                self.path = Folder(source)
                self.single = False
                example_file: File = self.path.children[0]
                if example_file.extension==".h5": self._from_folder_of_h5(self.path)
            #otherwise its a file
            case File() | str() if isfile(source):
                self.path = File(source)
                self.single = True
                if self.path.extension=="h5": self._from_h5(self.path)
                else: self._from_csv(self.path)
            #or if its already been read
            case Array():
                self.single = True
                self._from_numpy(array(source))
            case _: raise TypeError(f"{source} of wrong type: {type(source)}")
        # read the stats file
        if parent: self.stats = Folder(f"{self.parent.path}/stats")
        if self.stats: self._read_stats()
    def __len__(self) -> int: return 1 if self.single else len(self.file_names)
    def __iter__(self) -> Self:
        assert not self.single, "Cannot iterate through single scalar field"
        self.index = 0
        return self
    def __next__(self) -> NDArray:
        if self.index < len(self):
            i = self.index
            self.index += 1
            return self[i]
        else: raise StopIteration
    def __getitem__(self, item: int|slice|ArrayLike) -> NDArray:
        if self.single: return self.array[item]
        match item:
            case int():
                return self.cache[item] if self.caching and item in self.cache \
                       else self.reader(self.file_names[item], item)
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0,
                        item.stop if not item.stop is None else len(self),
                        item.step if not item.step is None else 1
                    )
                ]
                return array([
                    self.cache[i] if self.caching and i in self.cache \
                    else self.reader(self.file_names[i], i) for i in item_iters
                ])
            case Array(): return array([
                self.cache[i] if self.caching and i in self.cache \
                else self.reader(self.file_names[i], i) for i in item
            ])
    def _from_folder_of_h5(self, path: Folder) -> None:
        self.file_names: list = sorted(self.path.glob("*.h5"))
        self.reader: Callable = self._read_h5_file
        self.shape = self[0].shape
        self.ndims = len(self.shape)
        self.size = prod(self.shape) * len(self)
    def _from_h5(self, file:str) -> None:
        self.file = file.path if isinstance(file, File) else file
        self.array = self._read_h5_file(file, 0)
        self.shape = self.array.shape
        self.ndims = len(self.shape)
        self.size = prod(self.shape) * len(self)
    def _read_h5_file(self, file:str, item) -> NDArray:
        with h5File(file, 'r') as f:
            output = array(f["DATA"][:])
            #GODDMANIT I HATE THAT IT DOES Y,X and not X,Y
            if self.caching: self.cache[item] = output
            return output
    def _from_csv(self, source) -> None:
        ...
    def _read_csv_file(self, file:str) -> None: pass
    def _from_numpy(self, arr:ndarray) -> None:
        self.single = True
        self.array = arr
        self.shape = arr.shape
        self.ndims = len(self.shape)
    def _read_stats(self) -> None:
        ensure_path(self.stats.path)
        self.statsFile = File(f"{self.stats.path}/{self.name}.csv")
        if self.statsFile.exists:
            self.log.info(f"READING STATSFILE: {self.name}")
            (
                self.stats.min,
                self.stats.max,
                self.stats.median,
                self.stats.mean,
                self.stats.std
            ) = loadtxt(
                self.statsFile.path, delimiter=',', dtype=float32, skiprows=1
            )
            self.min = nanmin(self.stats.min)
            self.max = nanmax(self.stats.max)
        else:
            self.log.warning("CREATING STATSFILE, THIS MAY TAKE A WHILE...")
            ensure_path(self.statsFile.parent.path)
            data = array([
                [nanmin(frame) for frame in self],
                [nanmax(frame) for frame in self],
                [nanmean(frame) for frame in self],
                [nanmedian(frame) for frame in self],
                [nanstd(frame) for frame in self],
            ], dtype=float32)
            self.min = nanmin(data[0])
            self.max = nanmax(data[1])
            savetxt(self.statsFile.path, data, delimiter=',', header="min,max,median,mean,std")
    @cached_property #cached property only is used if self.min not set by stats file
    def min(self) -> Number:
        """docstring"""
        x = inf
        for frame in self:
            y = append(frame, x)
            x = nanmin(y)
        return x
    @cached_property #cached property only is used if self.max not set by stats file
    def max(self) -> Number:
        """docstring"""
        x = -inf
        for frame in self:
            y = append(frame, x)
            x = nanmax(y)
        return x
    @cached_property
    def extrema(self) -> tuple[Number]:
        """docstring"""
        return self.min, self.max
    def show(self, item:int, **kwargs) -> None:
        """docstring"""
        if hasattr(self.parent, 'dx'):
            x_ticks = arange(0, self.parent.input.boxsize[0], self.parent.dx)
            y_ticks = arange(0, self.parent.input.boxsize[1], self.parent.dy)
            show(self[item], x=x_ticks, y=y_ticks, **kwargs)
        assert False
    def movie(self, file_name=None, norm='none', cmap=default_cmap, func=None,**kwds) -> None:
        """docstring"""
        show_video(
            self[:] if func is None else [func(x) for x in self],
            self.name if file_name is None else file_name,
            norm=norm, cmap=cmap, **kwds
        )
