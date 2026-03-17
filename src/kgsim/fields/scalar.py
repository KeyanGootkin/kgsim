# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kbasic.parsing import File, Folder, ensure_path
from kbasic.typing import Array, ArrayLike, Number
from kbasic.strings import purple
from kplot import show, default_cmap, show_video
from functools import cached_property
from os import system
from numpy.typing import NDArray
from numpy import ndarray, array, prod, arange, nanmin, nanmax, nanmean, nanstd, \
                  nanmedian, inf, append, float32, loadtxt, savetxt
from os.path import isdir, isfile 
from glob import glob
from typing import Optional, Self, Any
from h5py import File as h5File

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
        source: str|Folder|File|Array, 
        parent: Optional = None,
        stats: Optional[Folder | str] = None,
        caching: bool = False,
        verbose: bool = False,
        name: Optional[str] = None, 
        latex: Optional[str] = None
    ) -> None:
        self.name: str = name 
        self.latex: str = latex
        self.verbose: bool = verbose
        self.parent = parent
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
            case x if type(x) in Array.types:
                self.single = True
                self._from_numpy(array(source))
            case _: raise TypeError(f"{source} of wrong type: {type(source)}")
        # read the stats file
        if not parent is None: self.stats = Folder(f"{self.parent.path}/stats")
        if not self.stats is None: self._read_stats()
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
                return self.cache[item] if self.caching and item in self.cache.keys() else self.reader(self.file_names[item], item)
            case slice():
                item_iters = [
                    i for i in range(
                        item.start if not item.start is None else 0, 
                        item.stop if not item.stop is None else len(self), 
                        item.step if not item.step is None else 1
                    )
                ]
                return array([
                    self.cache[i] if self.caching and i in self.cache.keys() else self.reader(self.file_names[i], i) for i in item_iters
                ])
            case x if type(x) in Array.types: return array([
                self.cache[i] if self.caching and i in self.cache.keys() else self.reader(self.file_names[i], i) for i in item
            ])
    def _from_folder_of_h5(self, path:str) -> None: 
        self.path = path.path if isinstance(path, Folder) else path
        self.file_names: list = sorted(glob(path + "/*.h5"))
        self.reader: function = self._read_h5_file
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
    def _from_csv(self) -> None:
        self.single = True
    def _read_csv_file(self) -> None: pass
    def _from_numpy(self, array:ndarray) -> None:
        self.single = True
        self.array = array
        self.shape = array.shape
        self.ndims = len(self.shape)
    def _read_stats(self) -> None:
        ensure_path(self.stats.path)
        self.statsFile = File(f"{self.stats.path}/{self.name}.csv")
        if self.statsFile.exists:
            if self.verbose: print(purple(f"READING STATSFILE: {self.name}"))
            self.stats.min, self.stats.max, self.stats.median, self.stats.mean, self.stats.std = loadtxt(
                self.statsFile.path, delimiter=',', dtype=float32, skiprows=1
            )
            self.min = nanmin(self.stats.min)
            self.max = nanmax(self.stats.max)
        else:
            if self.verbose: print(purple(f"CREATING STATSFILE: {self.name}"))
            ensure_path(self.statsFile.parent.path)
            data = array([
                [nanmin(frame) for frame in self],
                [nanmax(frame) for frame in self],
                [nanmean(frame) for frame in self],
                [nanmedian(frame) for frame in self],
                [nanstd(frame) for frame in self],
            ], dtype=float32)
            self.min = nanmin(data[:,0])
            self.max = nanmax(data[:,1])
            savetxt(self.statsFile.path, data, delimiter=',', header="min,max,median,mean,std")
    @cached_property #cached property only is used if self.min not set by stats file
    def min(self) -> Number:
        x = inf 
        for frame in self:
            y = append(frame, x)
            x = nanmin(y)
        return x
    @cached_property #cached property only is used if self.max not set by stats file
    def max(self) -> Number:
        x = -inf 
        for frame in self:
            y = append(frame, x)
            x = nanmax(y)
        return x
    @cached_property
    def extrema(self) -> tuple[Number]:
        return self.min, self.max
    def show(self, item:int, **kwargs) -> None: 
        if hasattr(self.parent, 'dx'):
            x_ticks = arange(0, self.parent.input.boxsize[0], self.parent.dx)
            y_ticks = arange(0, self.parent.input.boxsize[1], self.parent.dy)
        show(self[item], x=x_ticks, y=y_ticks, **kwargs)
    def movie(self, norm='none', cmap=default_cmap, func=None,**kwrg) -> None:
        @show_video(name=self.name, latex=self.latex, norm=norm, cmap=cmap)
        def reveal_thyself(s, func=func, **kwargs): 
            return array([self[i] for i in range(len(self))]) if alter_func is None else array([alter_func(self[i]) for i in range(len(self))])
        reveal_thyself(self if self.parent is None else self.parent, alter_func=alter_func,**kwrg)