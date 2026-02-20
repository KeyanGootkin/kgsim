from kbasic.parsing import File, Folder

thisFile = File(__file__)
pysimDir: Folder = thisFile.grandparent # Where the package lives
dHybridRtemplate: Folder = pysimDir + "/templates/dHybridR/" # where the base dHybridR template is