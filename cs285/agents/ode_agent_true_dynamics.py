from typing import Callable, Optional, Tuple, Sequence
import numpy as np
import gym
from cs285.infrastructure import pytorch_util as ptu
from cs285.agents.ode_agent import ODEAgent
from torchdiffeq import odeint
from tqdm import trange
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, Dopri5
import optax


@eqx.filter_jit
def pendulum_true_dynamics(self, t, y, args):
    times = args["times"] # (ep_len)
    actions = args["actions"] # (ep_len, ac_dim)
    idx = jnp.searchsorted(times, t, side="right") - 1
    action = actions[idx] # (ac_dim,)
    return #TODO: get the true dynamics of the pendulum environment


class ODEAgent_True_Dynamics(ODEAgent):
    def __init__(
        self,
        env: gym.Env,
        true_dynamics: Callable, # e.g. pendulum_true_dynamics
        mpc_horizon_steps: int,
        mpc_discount: float,
        mpc_timestep: float,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__(
            env=env,
            key=jax.random.PRNGKey(0), # just for convenience
            hidden_size=16, # just for convenience
            num_layers=1, # just for convenience
            ensemble_size=1, # just for convenience
            train_timestep=None,
            mpc_horizon_steps=mpc_horizon_steps,
            mpc_discount=mpc_discount,
            mpc_timestep=mpc_timestep,
            mpc_strategy=mpc_strategy,
            mpc_num_action_sequences=mpc_num_action_sequences,
            cem_num_iters=cem_num_iters,
            cem_num_elites=cem_num_elites,
            cem_alpha=cem_alpha,
        )
        self.ode_functions = None
        self.ensemble_size = None
        self.optims = None
        self.optim_states = None
        self.true_dynamics = true_dynamics
    
    def update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray, discount: float=1.0):
        pass

    def batched_update_gd(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray, discount: float=1.0):
        pass

    @eqx.filter_jit
    def evaluate_action_sequences(self, obs: jnp.ndarray, acs: jnp.ndarray, mpc_discount_arr: jnp.ndarray):
        times = jnp.linspace(0, (self.mpc_horizon_steps - 1) * self.mpc_timestep, self.mpc_horizon_steps)

        def evaluate_single_sequence(ac):
            ode_out = diffeqsolve(
                terms=diffrax.ODETerm(self.true_dynamics),
                solver=self.solver,
                t0=times[0],
                t1=times[-1],
                dt0=self.mpc_timestep,
                y0=obs,
                args={"times": times, "actions": ac},
                saveat=diffrax.SaveAt(ts=times)
            )
            rewards, _ = self.env.get_reward_jnp(ode_out.ys, ac)
            return jnp.mean(rewards * mpc_discount_arr)
        
        avg_rewards = jax.vmap(evaluate_single_sequence)(acs) # (seqs,)
        assert avg_rewards.shape == (self.mpc_num_action_sequences,)
        return avg_rewards