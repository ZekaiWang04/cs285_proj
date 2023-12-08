import numpy as np
import gym
from typing import Optional, Callable
from cs285.agents.ode_agent import ODEAgent
from cs285.envs.dt_sampler import BaseSampler
from tqdm import trange
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve

def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@eqx.filter_jit
def pendulum_true_dynamics(t, y, args):
    times = args["times"] # (ep_len)
    actions = args["actions"] # (ep_len, ac_dim)
    idx = jnp.searchsorted(times, t, side="right") - 1
    action = actions[idx] # (ac_dim,)
    
    cos_theta, sin_theta, thdot = y
    th = angle_normalize(jnp.arctan2(sin_theta, cos_theta))
    max_speed = 8
    max_torque = 2.0
    g = 10.0
    m = 1.0
    l = 1.0
    u = jnp.clip(action, -max_torque, max_torque)[0]
    newthdot = thdot + (3 * g / (2 * l) * sin_theta + 3.0 / (m * l**2) * u) * (times[idx+1] - times[idx])
    newthdot = np.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * (times[idx+1] - times[idx])
    return jnp.asarray([-jnp.sin(newth) * newthdot, jnp.cos(newth) * newthdot, 3 * g / (2 * l) * sin_theta + 3.0 / (m * l**2) * u])


class ODEAgent_True_Dynamics(ODEAgent):
    def __init__(
        self,
        env: gym.Env,
        mpc_horizon_steps: int,
        mpc_discount: float,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        mpc_dt_sampler: BaseSampler,
        mpc_timestep: float,
        true_dynamics: Callable = pendulum_true_dynamics, # e.g. pendulum_true_dynamics
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__(
            env=env,
            key=jax.random.PRNGKey(0), # dummy
            mlp_dynamics_setup={"hidden_size": 32, "num_layers": 4, "activation": "relu", "output_activation": "identity"}, # dummy
            optimizer_name="adamw", # dymmy
            optimizer_kwargs={"lr": 1e-3}, # dummy
            ensemble_size=1, # dummy
            train_timestep=None, # dummy
            train_discount=1.0, # dymmy
            mpc_horizon_steps=mpc_horizon_steps,
            mpc_discount=mpc_discount,
            mpc_strategy=mpc_strategy,
            mpc_num_action_sequences=mpc_num_action_sequences,
            mpc_dt_sampler=mpc_dt_sampler,
            mpc_timestep=mpc_timestep,
            cem_num_iters=cem_num_iters,
            cem_num_elites=cem_num_elites,
            cem_alpha=cem_alpha,
        )
        self.ode_functions = None
        self.optims = None
        self.optim_states = None
        self.true_dynamics = true_dynamics
    
    def update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        pass

    def batched_update_gd(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        pass

    @eqx.filter_jit
    def evaluate_action_sequences(self, obs: jnp.ndarray, acs: jnp.ndarray, mpc_discount_arr: jnp.ndarray):
        dts = self.mpc_dt_sampler.get_dt(size=(self.mpc_horizon_steps,))
        times = jnp.cumsum(dts) # (self.mpc_horizon_steps, )

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