from kbasic.parsing import File, Folder

thisFile = File(__file__)
kgsimDir: Folder = thisFile.parent.parent # Where the package lives
dHybridRtemplate: Folder = kgsimDir + "/templates/dHybridR/" # where the base dHybridR template is