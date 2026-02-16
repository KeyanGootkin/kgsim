class Species:
    def __init__(
            self,
            name: str,
            m: float = 1.,
            q: float = 1.
        ):
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
            index: int
    ):
        self.species = species 
        self.index = index 
        
    def __repr__(self): return f"{self.species}: {self.index}"