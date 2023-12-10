from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import gym
import jax
from typing import Optional, Sequence
from cs285.envs.dt_sampler import BaseSampler, ConstantSampler, UniformSampler, ExponentialSampler

default_mlp_setup = {
    "hidden_size": 32,
    "num_layers": 4,
    "activation": "relu",
    "output_activation": "identity",
}

default_neural_ode_kwargs = {
    "ode_dt0": 0.005, 
    "neural_ode_kwargs": default_mlp_setup
}

def ode_config(
    env_name: str,
    exp_name: str,
    key: int = 0,
    dt_sampler_name: str = "constant",
    dt_sampler_kwargs: dict={"dt": 0.05},
    neural_ode_name: str = "vanilla",
    neural_ode_kwargs: dict = default_neural_ode_kwargs,
    optimizer_name: str = "adamw",
    optimizer_kwargs: dict = {"lr": 1e-3},
    ensemble_size: int = 3,
    mpc_horizon_steps: int = 100,
    mpc_discount: float = 1.0,
    mpc_strategy: str = "random",
    mpc_num_action_sequences: int = 1000,
    mpc_dt_sampler_name: str = "constant",
    mpc_dt_sampler_kwargs: dict = {"dt": 0.05},
    cem_num_iters: Optional[int] = None,
    cem_num_elites: Optional[int] = None,
    cem_alpha: Optional[float] = None,
    initial_trajs: int = 100,  # number of trajectories to collect with random policy at the start
    trajs: int = 100,  # number of transitions to collect per per iteration thereafter
    train_batch_size: int = 64,  # number of transitions to train each dynamics model per iteration
    num_iters: int = 20,
    replay_buffer_capacity: int = 1000,
    num_agent_train_steps_per_iter: int = 100,
    num_eval_trajectories: int = 10,
    train_discount: float = 1.0,
    train_ep_len: int=200,
    train_stride: int=1
):
    key = jax.random.PRNGKey(key)
    agent_key, dt_sampler_key, mpc_dt_sampler_key = jax.random.split(key, 3)
    dt_sampler_kwargs["key"] = dt_sampler_key
    mpc_dt_sampler_kwargs["key"] = mpc_dt_sampler_key
    # hardcoded for this assignment
    if env_name == "pendulum-cs285-v0":
        ep_len = 200
    
    assert train_ep_len <= ep_len

    dt_sampler = {"constant": ConstantSampler,
                  "uniform": UniformSampler,
                  "exponential": ExponentialSampler}[dt_sampler_name](**dt_sampler_kwargs)
    def make_env(render: bool = False):
        return RecordEpisodeStatistics(
            gym.make(env_name, 
                     dt_sampler=dt_sampler,
                     render_mode="single_rgb_array" if render else None),
        )
    
    mpc_dt_sampler = {"constant": ConstantSampler,
                      "uniform": UniformSampler,
                      "exponential": ExponentialSampler}[mpc_dt_sampler_name](**mpc_dt_sampler_kwargs)

    log_string = f"{env_name}_{exp_name}"

    return {
        "agent_kwargs": {
            "key": agent_key,
            "neural_ode_name": neural_ode_name,
            "neural_ode_kwargs": neural_ode_kwargs,
            "optimizer_name": optimizer_name,
            "optimizer_kwargs": optimizer_kwargs,
            "ensemble_size": ensemble_size,
            "train_discount": train_discount,
            "mpc_horizon_steps": mpc_horizon_steps,
            "mpc_discount": mpc_discount,
            "mpc_strategy": mpc_strategy,
            "mpc_num_action_sequences": mpc_num_action_sequences,
            "mpc_dt_sampler": mpc_dt_sampler,
            "cem_num_iters": cem_num_iters,
            "cem_num_elites": cem_num_elites,
            "cem_alpha": cem_alpha,
        },
        "make_env": make_env,
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "num_iters": num_iters,
        "ep_len": ep_len,
        "trajs": trajs,
        "initial_trajs": initial_trajs,
        "train_batch_size": train_batch_size,
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "num_eval_trajectories": num_eval_trajectories,
        "train_ep_len": train_ep_len,
        "train_stride": train_stride
    }
