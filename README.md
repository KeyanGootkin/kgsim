# PySim
PySim is a package for simply interacting with simulations in a pythonic, object oriented way.

```python
from pysim.dhybridr.turbulence import TurbSim

s = TurbSim("path/to/simulation")
# examine initial conditions
s.B.z.show(0)
s.u.x.show(0)
# make video of density evolution over simulation
from kplot import show_video
@show_video(name='energy_flux', latex=r'$\rho \mathcal{u}_\perp^2$')
def energy_flux(s, **kwargs) -> np.ndarray:
    return np.array([p*(ux**2+uy**2) for p, ux, uy in zip(s.density, s.u.x, s.u.y)])

energy_flux(s)
```

