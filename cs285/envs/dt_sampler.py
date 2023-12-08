import jax.numpy as jnp
import jax

class BaseSampler():
    def __init__(self, **kwargs):
        pass

    def get_dt(self, size=()):
        pass

class ConstantSampler(BaseSampler):
    def __init__(self, dt: float, key: jax.random.PRNGKey=None):
        super().__init__()
        self.dt = dt

    def get_dt(self, size=()):
        return self.dt * jnp.ones(shape=size)
    
class UniformSampler(BaseSampler):
    def __init__(self, high: float, low: float, key: jax.random.PRNGKey):
        super().__init__()
        self.high = high
        self.low = low
        self.key = key

    def get_dt(self, size=()):
        self.key, sample_key = jax.random.split(self.key)
        return jax.random.uniform(key=sample_key, shape=size, minval=self.low, maxval=self.high)

class ExponentialSampler(BaseSampler):
    def __init__(self, lmbda: float, key: jax.random.PRNGKey):
        super().__init__()
        self.lmbda = lmbda
        self.key = key

    def get_dt(self, size=()):
        self.key, sample_key = jax.random.split(self.key)
        return jax.random.exponential(key=sample_key, shape=size) / self.lmbda
