from kgsim.parsing import Folder
user = __file__.split("/")[1]
simulationDir = Folder(f"/Users/{user}/sims/")
figDir = Folder(f"/Users/{user}/figures/")
frameDir = Folder(f"/Users/{user}/frames/")
videoDir = Folder(f"/Users/{user}/videos/")