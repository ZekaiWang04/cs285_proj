import numpy as np

class BaseSampler():
    def __init__(self, **kwargs):
        pass

    def get_dt(self, size=()):
        pass

class ConstantSampler(BaseSampler):
    def __init__(self, dt: float, seed=0):
        super().__init__()
        self.dt = dt

    def get_dt(self, size=()):
        return self.dt * np.ones(shape=size)
    
class UniformSampler(BaseSampler):
    def __init__(self, high: float, low: float=0.0, seed=0):
        super().__init__()
        self.high = high
        self.low = low
        self.rng = np.random.default_rng(seed)

    def get_dt(self, size=()):
        return self.rng.uniform(low=self.low, high=self.high, size=size)

class ExponentialSampler(BaseSampler):
    def __init__(self, lmbda: float, seed=0):
        super().__init__()
        self.lmbda = lmbda
        self.rng = np.random.default_rng(seed)

    def get_dt(self, size=()):
        return self.rng.exponential(scale=1/self.lmbda, size=size)
