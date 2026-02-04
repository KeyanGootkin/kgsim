# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
#pysim imports
from pysim.utils import nan_clip
from pysim.parsing import File, Folder, ensure_path
from pysim.environment import frameDir, videoDir
#nonpysim imports
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from functools import wraps
import os
from matplotlib.colors import LinearSegmentedColormap, ListedColormap,hex2color
from h5py import File as h5File
from tqdm import tqdm

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
pink = "#E34F68"
lightpink = "#E39FAA"
blue = "#7350E6"
lightblue = "#AE9FE3"
shadow = "#B8B7B8"
manoaskies = LinearSegmentedColormap.from_list("manoaskies", [pink, blue])
manoaskies_centered = LinearSegmentedColormap.from_list("manoaskies_centered", [lightpink, pink, "#000000", blue, lightblue])
manoaskies_background_blue = "#0C0524"
pink2grey = LinearSegmentedColormap.from_list("p2g", [pink, shadow])
grey2black = LinearSegmentedColormap.from_list("g2b", [shadow, "#000000"])
colors_list = np.zeros((256, 4))
colors_list[:128] =  list(hex2color(manoaskies_background_blue))+[1]
for i in range(28): colors_list[128+i] = pink2grey(i/28)
for i in range(100): colors_list[156+i] = grey2black(i/150)
manoaskies_beauty = ListedColormap(colors_list)
default_cmap = plt.cm.plasma

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# # Ploting utils
def auto_norm(
    norm: str, 
    frames: np.ndarray, 
    linear_threshold: float|None = None, 
    center: float|None = None, 
    saturate: float|None = None
):
    """A function to create a matplotlib normalization given a set images.

    Args:
        norm (str): what type of scale to use, e.g. lognorm or centerednorm
        frames (np.ndarray): the images to base the normalization on
        linear_threshold (float | None, optional): for symlognorm. Defaults to None.
        center (float | None, optional): for centered normalizations. Defaults to None.
        saturate (float | None, optional): the level at which to saturate the norm, e.g. if saturate=0.01 then the 
            max is the 99th percentile. Defaults to None.

    Returns:
        matplotlib normalization
    """
    frames = frames[(-np.inf < frames)&(frames < np.inf)]
    # set min/max IF saturate is None                          or IF saturate is a tuple                                          ELSE assume its a float
    low = np.nanmin(frames) if saturate is None else np.nanquantile(frames, 1-saturate[0]) if isinstance(saturate, tuple) else np.nanquantile(frames, 1-saturate)
    high = np.nanmax(frames) if saturate is None else np.nanquantile(frames, 0+saturate[1]) if isinstance(saturate, tuple) else np.nanquantile(frames, 0+saturate)
    match norm.lower():
        case "lognorm"|"log":
            if low < 0: raise ValueError(f"minimum is {low}, LogNorm only takes positive values")
            if low==0: low=np.nanmin(frames[frames!=0])
            return LogNorm(vmin=low, vmax=high)
        case "symlognorm"|"symlog"|"sym":
            sig = np.nanstd(frames)
            mu = np.nanmean(frames)
            if np.abs(mu)-sig > 0: raise TypeError("SymLogNorm is only designed for stuff close to zero!")
            return SymLogNorm(sig if linear_threshold is None else linear_threshold, vmin=low, vmax=high)
        case n if n in ["centerednorm", "twoslope", "twoslopenorm"]:
            sig = np.nanstd(frames)
            mu = np.nanmean(frames)
            # for the center use center if give otherwise use 0 if mean is small, else use mean
            vcenter = center if not center is None else 0 if np.abs(mu)-sig > 0 else mu
            return TwoSlopeNorm(vmin=low, vcenter=vcenter, vmax=high)
        case _: return Normalize(vmin=low, vmax=high)
def tile(arr: np.ndarray) -> np.ndarray:
    """take an image and create a 3x3 grid of that image"""
    return np.r_[np.c_[arr, arr, arr], np.c_[arr, arr, arr], np.c_[arr, arr, arr]]
def align_algorithm(x: list|np.ndarray, mode: str):
    match mode:
        case 'mid'|'m'|'center'|'c':
            return [(x[i] + x[i+1])/2 for i in range(len(x)-1)]
        case 'logmid'|'lm':
            return [np.log10((10**x[i] + 10**x[i+1])/ 2) for i in range(len(x)-1)]
        case 'left'|'l':
            return x[:-1]
        case 'right'|'r':
            return x[1:]
# Plots
def show(
        field: np.ndarray,
        tile_image: bool = False,
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        cmap = default_cmap,
        colorbar: bool = True,
        colorbar_style: dict = {'location':'right', "size":"7%", "pad":0.05},
        cticks: list|None = None,
        units: str|None = None,
        #contour options
        contour: np.ndarray|None = None,
        contour_style: dict = {'levels': 10, 'colors': 'black'},
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #everything else goes into pcolormesh
        **kwargs
):
    """a convinient way to plt.imshow and image and arrange it nicely

    Args:
        field (np.ndarray): The image to be plotted
        tile_image (bool, optional): whether to tile the image. Defaults to False.
        x (np.ndarray | None, optional): x coordinates of pixels. Defaults to None.
        y (np.ndarray | None, optional): y coordinates of pixels. Defaults to None.
        fig (plt.Figure, optional): the figure on which to plot. Defaults to None.
        ax (plt.Axes, optional): the ax on which to plot this image. Defaults to None.
        axis (bool, optional): whether to include the whole axis, if False then only the plot area will be shown.
        figsize (tuple[float, float], optional): unless fig and ax are given plot on a figure of this size. Defaults to (10, 10).
        show (bool, optional): whether to use the plt.show() command at the end. Defaults to True.
        cmap (plt.ColorMap, optional): the colormap to use for the image. Defaults to plasma.
        colorbar (bool, optional): whether to add a colorbar. Defaults to True.
        colorbar_style (dict, optional): dictionary containing kwargs for the colorbar command. Defaults to {'location':'right', "size":"7%", "pad":0.05}.
        cticks (list | None, optional): the ticks to use on the colorbar. Defaults to None.
        units (str | None, optional): label for the colorbar. Defaults to None.
        contour (np.ndarray | None, optional): whether to put a contour plot on top. Defaults to None.
        contour_style (dict, optional): kwargs for contour plotting command. Defaults to {'levels': 10, 'colors': 'black'}.
        xlim (tuple, optional): the limits on the x axis. Defaults to (None, None).
        ylim (tuple, optional): the limits on the y axis. Defaults to (None, None).
        title (str | None, optional): the plot title. Defaults to None.
        save (str, optional): if given save the plot to this path. Defaults to "".
        dpi (int, optional): dots per inch to save at. Defaults to 100.

    Returns:
        fig (plt.Figure): The figure on which the plot was put
        ax (plt.Axes): The ax on which the plot was put
        img (plt.Artist): the plot artist
    """
    # prep data
    assert (ndims:=len(field.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    image = tile(field) if tile_image else field
    # check axes
    if x is None: x, y = np.mgrid[:image.shape[0], :image.shape[1]]
    if tile_image: x, y = np.r_[x, x+x[-1], x+2*x[-1]], np.r_[y, y+y[-1], y+2*y[-1]]
    else: assert (len(x), len(y)) == image.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {image.shape}"
    # prep figure
    if ax is None: (fig, ax) = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    if not axis: 
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
        colorbar = False
    # plot data
    img = ax.pcolormesh(x, y, image, cmap=cmap, **kwargs)
    # colorbar
    if colorbar: 
        divider = make_axes_locatable(ax)
        colorbar_location = colorbar_style.pop("location") if "location" in colorbar_style.keys() else "right"
        cax = divider.append_axes(colorbar_location, **colorbar_style)
        fig.colorbar(img, cax=cax, ax=ax, ticks=cticks, label=units, orientation = 'vertical' if colorbar_location in ('right', 'left') else 'horizontal')
        if colorbar_location=='top':
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')
    # contour
    if not contour is None: ax.contour(contour, **contour_style)
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    return fig, ax, img
def contour(
        Z: np.ndarray, 
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        levels = 10,
        color = 'black',
        colors = None,
        cmap = None,
        negative_linestyles = '-',
        linestyles = '-',
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #throw the rest in a dict
        **kwds
    ):
    # prep data
    assert (ndims:=len(Z.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    # check axes
    if x is None: x, y = np.mgrid[:Z.shape[0], :Z.shape[1]]
    else: assert (len(x), len(y)) == Z.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {image.shape}"
    # prep figure
    if ax is None: (fig, ax) = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    if not axis: 
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
        colorbar = False
    # plot data
    contour_dict = {'levels':levels,"linestyles":linestyles,'negative_linestyles':negative_linestyles}
    if cmap: #plot with a cmap
        contour_dict['cmap'] = cmap
    else: #plot with a color
        contour_dict['colors'] = color if not colors else colors
    img = ax.contour(
        x, y, Z,
        **contour_dict,
        **kwds
    )
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    return fig, ax, img
def contourf(
        Z: np.ndarray, 
        #x/y axes
        x: np.ndarray|None = None,
        y: np.ndarray|None = None,
        #figure setup
        fig = None,
        ax = None,
        axis: bool = True,
        figsize: tuple[float, float] = (10, 10),
        show: bool = False,
        #plot parameters
        levels = 10,
        cmap = default_cmap,
        colorbar: bool = True,
        colorbar_style: dict = {'location':'right', "size":"7%", "pad":0.05},
        cticks: list|None = None,
        units: str|None = None,
        #plot formating
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str|None = None,
        #presentation parameters
        save: str = "",
        dpi: int = 100,
        #throw the rest in a dict
        **kwds
    ):
    # prep data
    assert (ndims:=len(Z.shape))==2, f"show was given an image with {ndims} dimensions, please provide a 2d array"
    # check axes
    if x is None: x, y = np.mgrid[:z.shape[0], :z.shape[1]]
    else: assert (len(x), len(y)) == z.shape, f"Given x of shape {len(x)} and y of shape {len(y)} but image is of shape {image.shape}"
    # prep figure
    if ax is None: (fig, ax) = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    if not axis: 
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])
        colorbar = False
    # plot data
    contour_dict = {'levels':levels,"cmap":cmap}
    img = ax.contourf(
        x, y, Z,
        **contour_dict,
        **kwds
    )
    # colorbar
    if colorbar: 
        divider = make_axes_locatable(ax)
        colorbar_location = colorbar_style.pop("location") if "location" in colorbar_style.keys() else "right"
        cax = divider.append_axes(colorbar_location, **colorbar_style)
        fig.colorbar(img, cax=cax, ax=ax, ticks=cticks, label=units)
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set title
    ax.set_title(title)
    # set aspect
    ax.set_aspect('equal')
    # present the figure
    if len(save)>0: 
        if not '.' in save: save +=".jpeg"
        plt.savefig(save, dpi=dpi)
    if show: plt.show()
    return fig, ax, img
def diffplot(x, y, align: str = 'mid', **kwargs):
    dx = np.diff(x)
    dy = np.diff(y)
    x_aligned = align_algorithm(x, align)
    return plt.plot(x_aligned, dy/dx, **kwargs)
# Videos
def plot_video(
    xs, ys, 
    fname, 
    ax=None,
    fps=10, 
    figsize=(5, 5), 
    **kwargs
):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()
    [line] = ax.plot(xs[0], ys[0], **kwargs)
    def update(f: int):
        line.set_xdata(xs[f])
        line.set_ydata(ys[f])
    anim = FuncAnimation(fig, update, frames=len(xs))
    anim.save(fname, fps=fps)
def show_video(
    frames: np.ndarray,
    fname: str,
    ax = None,
    norm = 'linear',
    fps: int = 30,
    **kwargs
):
    # create figure and axis 
    fig,ax,img = show(frames[0], ax=ax, norm=auto_norm(norm, frames), show=False, **kwargs)
    def update(f: int):
        img.set_array(frames[f])
    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(fname, fps=fps)
def particle_video(
    sim,
    particles: list[str] | int,
    background = None,      
    resume: bool = False,
    res: int = None,
    zfill: int = 10,
    dpi: int = 250,
    # paticle plotting keywords
    color = 'red',
    marker='.',
    ms=10, 
    # assume the rest are keywords for the show function
    **kwds
):
    # check species
    assert sim.input.num_species == 1, "only one species is implemented rn :-("
    # check the time resolution is valid
    if not res: res = sim.input.sp01.track_nstore
    dnp = sim.input.sp01.track_nstore
    dnb = sim.input.ndump
    assert (dnp % res == 0) & (res % dnb == 0), "your resolution must be an integer multiple of track_nstore and ndump must be an integer multiple of res."
    # handle the resuming
    if not resume: 
        os.system(f'rm {frameDir.path}/*')
        i_start = 0
    else:
        last_file = sorted(frameDir.children)[-1]
        last_iter = int(last_file[:-4])
        i_start = last_iter + res
    # find the file
    fn = sim.path+"/Output/Tracks/Sp01/track_Sp01.h5"
    with h5File(fn) as file:
        # extract particle tracks based on the particles argument
        tags = np.array(list(file.keys()))
        match particles:
            case [str(x), ]: selected = particles 
            case int(x): selected = np.random.choice(tags, size=x, replace=False)
        sx, sy = np.array([file[t]['x1'] for t in selected]).T, np.array([file[t]['x2'] for t in selected]).T 
        # setup background
        if not background: background = sim.density
        # initialize plots
        fig,ax,img = show(
            background[0], 
            x=range(0, sim.input.boxsize[0], sim.dx), y=range(0, sim.input.boxsize[1], sim.dy), 
            zorder=1,
            **kwds
        )
        line, = ax.plot(
            sx[0], sy[0], 
            ls='None', color=color, marker=marker, ms=ms,
            zorder = 2
        )
        # execute the loop
        for i in tqdm(range(i_start, sim.input.niter, res)):
            pind = i // dnp 
            line.set_data(sx[pind], sy[pind])
            if i % dnb == 0:
                bind = i // dnb 
                img.set_array(background[bind])
            plt.savefig(f"{frameDir.path}/{str(i).zfill(zfill)}.png", dpi=dpi)

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Decorators                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def plot_video_function(func):
    @wraps(func)
    def line_video_wrapper(*args, ax=None, save="default.gif", **kwargs):
        # Calculate data via func
        xs, ys = func(*args, **kwargs)
        plot_video(xs, ys, save, ax=ax)
    return line_video_wrapper
def show_video_function(func):
    @wraps(func)
    def simple_video_wrapper(*args, save="default.gif", norm='linear', **kwargs):
        frames = func(*args, **kwargs)
        show_video(frames, save, norm=norm)
    return simple_video_wrapper