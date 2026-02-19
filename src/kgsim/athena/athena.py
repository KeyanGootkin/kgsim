# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.fields import ScalarField, VectorField
from kgsim.simulation import GenericSimulation
from kgsim.athena.athena_read import athdf

from kbasic.parsing import Folder
from glob import glob 
import numpy as np
import builtins

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def get_mhd_config(mhd_run_dir, config_name):
    """Get MHD run information from the configuration file

    Arguments:
        mhd_run_dir (string): MHD run directory
        config_name (string): MHD simulation configuration file name
    """
    with open(mhd_run_dir + "/" + config_name) as f:
        contents = f.readlines()
    f.close()
    mhd_config = {}
    for line in contents:
        if "<" in line and ">" in line and "<" == line[0]:
            block_name = line[1:line.find(">")]
            mhd_config[block_name] = {}
        else:
            if line[0] != "#" and "=" in line:
                line_splits = line.split("=")
                tail = line_splits[1].split("\n")
                data = tail[0].split("#")
                ltmp = line_splits[0].strip()
                try:
                    mhd_config[block_name][ltmp] = float(data[0])
                except ValueError:
                    mhd_config[block_name][ltmp] = data[0].strip()
    return mhd_config

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class AthenaParameter(ScalarField):
    def __init__(
        self, 
        param_key: str,
        parent = None,
        caching: bool = False,
        verbose: bool = False,
        name: str = None, 
        latex: str = None
    ):
        super().__init__( 
            None, 
            parent=parent, 
            caching=caching, 
            verbose=verbose, 
            name=name, 
            latex=latex
        )
        self.param_key = param_key
        self.path = parent.path
        self.single = False 
        self.file_names = parent.file_names

    def reader(self, fname: str, item: int):
        return athdf(fname, quantities=[self.param_key])[self.param_key][0]

class Athena(GenericSimulation):
    def __init__(
            self, 
            path, 
            template = None, 
            caching = False, 
            verbose = True
    ):
        super().__init__(path, template, caching, verbose)
        self.input = get_mhd_config(self.path, "athinput.reconnection")
        self.parse_output()

    def parse_output(self):
        self.file_names = sorted(glob(self.path+"/*.athdf"))
        time, x, y, dxm, dym = [],[],[],[],[]
        for file in self.file_names:
            fdata = athdf(file, quantities=[])
            time.append(fdata['Time'])
            x.append(fdata["x1f"])
            y.append(fdata["x2f"])
            dxm.append(fdata["x1f"][1] - fdata["x1f"][0])
            dym.append(fdata["x2f"][1] - fdata["x2f"][0])
        self.time = np.array(time)
        self.dx = dxm[0]
        self.dy = dym[0]
        kwargs = {'caching':self.caching, 'verbose':self.verbose, 'parent':self}
        self.B = VectorField(
            AthenaParameter("Bcc1", name="B_x", latex="$B_x$", **kwargs),
            AthenaParameter("Bcc2", name="B_y", latex="$B_y$", **kwargs),
            AthenaParameter("Bcc3", name="B_z", latex="$B_z$", **kwargs),
            name='magnetic', latex=r"$\vec{B}$", **kwargs
        )
        self.u = VectorField(
            AthenaParameter("vel1", name="u_x", latex="$u_x$", **kwargs),
            AthenaParameter("vel2", name="u_y", latex="$u_y$", **kwargs),
            AthenaParameter("vel3", name="u_z", latex="$u_z$", **kwargs),
            name='velocity', latex=r"$\vec{u}$", **kwargs
        )
        self.density = AthenaParameter("rho", name='density', latex=r"$\rho$", **kwargs)
        self.P = AthenaParameter("press", name="pressure", latex="$P$", **kwargs)