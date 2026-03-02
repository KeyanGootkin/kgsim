# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
# >-|===|>                             Classes                             <|===|-<
# !==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==!==
class Species:
    def __init__(
            self,
            name: str,
            m: float = 1.,
            q: float = 1.
        ):
        """docstring"""
        self.name = name
        self.m = m 
        self.q = q 
        self.mtq = m/q # mass to charge ratio
        self.qtm = q/m # charge to mass ratio 
        
    def __repr__(self): return self.name
class Particle:
    def __init__(
            self,
            species: Species,
            tag: str
        ):
        """docstring"""
        self.species = species 
        self.tag = tag
        
    def __repr__(self): return f"{self.species.name}: {self.tag}"