# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from kgsim.parsing import Folder

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
anvil_user = __file__.split("/")[2]
simulationDir = Folder(f"/anvil/scratch/{anvil_user}/sims/") # default location to put/look for simulations
figDir = Folder(f"/home/{anvil_user}/turbulence/figures/") # default location for figures to go
frameDir = Folder(f"/home/{anvil_user}/frames/") # default location for frames of videos to go
videoDir = Folder(f"/home/{anvil_user}/videos/") # default location for videos to go
