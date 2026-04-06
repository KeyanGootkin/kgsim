# pylint: skip-file
import pytest
from kgsim.particles import Species, Particle
from kgsim.simulation import GenericSimulation
from kbasic import Vector
from kplot import plot, LineCollection

import numpy as np
import matplotlib.pyplot as plt

def test_main():
    x = np.arange(0, 10, .01)
    y = np.array(list(x/2)+list(x))
    energy = np.sin(np.arange(2*len(x)))
    p = Particle(x=list(x)*2, y=y)
    s = GenericSimulation('./', verbose=False)
    s.size = (10, 10)
    p.parent = s
    p.plot_trail(
        color=energy, norm='log', 
        save='./tests/output/periodic.png'
    )

    assert False

def test_particles():
    sp = Species('test')
    p = Particle(sp)
    assert p.species == sp
    assert sp.name=='test'