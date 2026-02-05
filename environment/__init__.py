# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Imports                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
from pysim.parsing import File, Folder

# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                           Definitions                           <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
thisFile = File(__file__)
pysimDir = Folder(thisFile.grandparent) # Where the package lives
dHybridRtemplate = pysimDir + "/templates/dHybridR/" # where the base dHybridR template is
# parse what kinda computer is running this
isAnvil: bool = Folder("/anvil").exists
isPC: bool = thisFile.path.lower().startswith("/users")
isWindows: bool = thisFile.path.lower().startswith(r"C:")
if isAnvil: 
    from pysim.environment.anvil import simulationDir # so shit don't complain
    from pysim.environment.anvil import *
elif isPC:
    user = thisFile.path.lower().split('/')[2]
    match user:
        case 'keyan': from pysim.environment.Keyan import *
        case _: from pysim.environment.defaultPC import *
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                            Functions                            <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
def available() -> list[str]: 
    return simulationDir.children