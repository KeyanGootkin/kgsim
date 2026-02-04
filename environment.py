# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from pysim.parsing import File, Folder 
"""
These are the environment variables that tells pysim where to find/look for different directories or templates or whatnot
"""

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
environment_variable_start_line = 17 #might have to change when you update this file
first_time_setup_line = 61
isAnvil: bool = Folder("/anvil").exists
isKeyan: bool = Folder("/Users/keyan").exists
thisFile = File(__file__)
simulationDir = Folder(f"/anvil/scratch/x-kgootkin/sims/") # default location to put/look for simulations
figDir = Folder(f"/home/x-kgootkin/turbulence/figures/") # default location for figures to go
frameDir = Folder(f"/home/x-kgootkin/frames/") # default location for frames of videos to go
videoDir = Folder(f"/home/x-kgootkin/videos/") # default location for videos to go
pysimDir = Folder(thisFile.parent) # Where the package lives
dHybridRtemplate = pysimDir + "/templates/dHybridR/" # where the base dHybridR template is

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def available() -> list[str]: 
    return simulationDir.children

def setup_environment():
    global isAnvil, isKeyan
    if isAnvil:
        anvil_user = pysimDir.path.split('/')[2]
        environment_variable_lines = [
            f'simulationDir = Folder("/anvil/scratch/{anvil_user}/sims/") # default location to put/look for simulations',
            f'figDir = Folder("/home/{anvil_user}/turbulence/figures/") # default location for figures to go',
            f'frameDir = Folder("/home/{anvil_user}/frames/") # default location for frames of videos to go',
            f'videoDir = Folder("/home/{anvil_user}/videos/") # default location for videos to go'
        ]
    elif isKeyan:
        environment_variable_lines = [
            'simulationDir = Folder("/Users/keyan/code/data/sims/")',
            'figDir = videoDir = Folder("./")',
            'frameDir = Folder("/Users/keyan/code/data/frames/")'
        ]
    else:
        environment_variable_lines = [
            f'simulationDir = Folder({input("please specify a default simulation directory: ")})',
            f'figDir = Folder({(i:=input("please specify a default directory for figures or enter nothing to use the files location: ")) if len(i)>0 else "."})',
            f'videoDir = Folder({(i:=input("please specify a default directory for videos or enter nothing to use the files location: ")) if len(i)>0 else "."})',
            f'framesDir = Folder({(i:=input("please specify a default directory for frames: "))})'
        ]
    exec("\n".join(environment_variable_lines), globals())
    old_lines = thisFile.lines 
    thisFile.lines[first_time_setup_line] = "FIRST_TIME_SETUP = False"
    thisFile.lines = old_lines[:environment_variable_start_line] + environment_variable_lines + old_lines[environment_variable_start_line:]
    thisFile.save(interactive=False)

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                               Run                               <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
FIRST_TIME_SETUP = False

if FIRST_TIME_SETUP: 
    setup_environment()