from typing import Callable, Optional, Tuple, Sequence
import numpy as np
import gym
from cs285.infrastructure import pytorch_util as ptu
from torchdiffeq import odeint
from tqdm import trange
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, Dopri5
import optax

class TrueDynamicsPendulum():
    def __init__():
        # TODO: do some thing
        pass

    @eqx.filter_jit
    def __call__(self, t, y, args):
        # TODO
        # do some thing, remember you can pass actions into args
        # see my ode_agent.py
        # use the true dynamics for pendulum env
        pass
    
class ODEAgent_Pendulum_True_Dynamics():
    def __init__(
        self,
        env: gym.Env,
        key: jax.random.PRNGKey,
        hidden_size: int,
        num_layers: int,
        ensemble_size: int,
        train_timestep: float,
        mpc_horizon_steps: int,
        mpc_timestep: float,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
        activation: str = "relu",
        output_activation: str = "identity",
        lr: float=0.001
    ):
        # super().__init__()
        self.env = env
        self.train_timestep = train_timestep
        self.mpc_horizon_steps = mpc_horizon_steps # in terms of timesteps
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        self.mpc_timestep = mpc_timestep # when evaluating

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        keys = jax.random.split(key, ensemble_size)
        self.ode_functions = [TrueDynamicsPendulum() for _ in range(ensemble_size)] # TODO
        self.optims = None # TODO
        self.optim_states = None # TODO

        self.solver = Dopri5() # TODO: maybe diffrax.Euler() or other solvers, see diffrax documentation
    
    def update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray, discount: float=1.0):
        pass

    def batched_update_gd(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray, discount: float=1.0):
        pass

    @eqx.filter_jit
    def evaluate_action_sequences(self, obs: jnp.ndarray, acs: jnp.ndarray):
        times = jnp.linspace(0, (self.mpc_horizon_steps - 1) * self.mpc_timestep, self.mpc_horizon_steps)

        def evaluate_single_sequnce(ac):
            avg_rewards = jnp.zeros((self.ensemble_size,))
            for i in range(self.ensemble_size):
                ode_func = self.ode_functions[i]
                ode_out = diffeqsolve(
                    terms=diffrax.ODETerm(ode_func),
                    solver=self.solver,
                    t0=times[0],
                    t1=times[-1],
                    dt0=self.mpc_timestep,
                    y0=obs,
                    args={"times": times, "actions": ac},
                    saveat=diffrax.SaveAt(ts=times)
                )
                rewards, _ = self.env.get_reward_jnp(ode_out.ys, ac)
                avg_rewards.at[i].set(jnp.mean(rewards))
            return jnp.mean(avg_rewards)
        
        avg_rewards = jax.vmap(evaluate_single_sequnce)(acs) # (seqs,)
        assert avg_rewards.shape == (self.mpc_num_action_sequences,)
        return avg_rewards

    @eqx.filter_jit
    def get_action(self, obs: jnp.ndarray, key: jax.random.PRNGKey):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        actions = jax.random.uniform(
            key=key,
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high,
            shape=(self.mpc_num_action_sequences, self.mpc_horizon_steps, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, actions)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = jnp.argmax(rewards)
            return actions[best_index, 0, :]
        elif self.mpc_strategy == "cem":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")