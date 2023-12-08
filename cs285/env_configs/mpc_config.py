from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import gym
from typing import Optional
from cs285.envs.dt_sampler import BaseSampler, UniformSampler, ConstantSampler, ExponentialSampler
import jax
import jax.numpy as jnp


def mpc_config(
    env_name: str,
    exp_name: str,
    key: int = 0,
    dt_sampler_name: str = "constant",
    dt_sampler_kwargs: dict={"dt": 0.05},
    learning_rate: float = 1e-3,
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
    initial_batch_size: int = 5000, 
    batch_size: int = 5000,
    num_iters: int = 20,
    replay_buffer_capacity: int = 1000,
    num_agent_train_steps_per_iter: int = 100,
    num_eval_trajectories: int = 10,
    hidden_size: int = 32,
    num_layers: int = 4,
    activation: str = "relu",
    output_activation: str = "identity",
    mode: str = "vanilla"
):
    key = jax.random.PRNGKey(key)
    # hardcoded for this assignment
    if env_name == "pendulum-cs285-v0":
        ep_len = 200

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

    log_string = f"{env_name}_{exp_name}_l{num_layers}_h{hidden_size}_mpc{mpc_strategy}_horizon{mpc_horizon_steps}_actionseq{mpc_num_action_sequences}"
    if mpc_strategy == "cem":
        log_string += f"_cem_iters{cem_num_iters}"

    return {
        "agent_kwargs": {
            "key": key,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "activation": activation,
            "output_activation": output_activation,
            "lr": learning_rate,
            "ensemble_size": ensemble_size,
            "mpc_horizon_steps": mpc_horizon_steps,
            "mpc_discount": mpc_discount,
            "mpc_strategy": mpc_strategy,
            "mpc_num_action_sequences": mpc_num_action_sequences,
            "mpc_dt_sampler": mpc_dt_sampler,
            "cem_num_iters": cem_num_iters,
            "cem_num_elites": cem_num_elites,
            "cem_alpha": cem_alpha,
            "mode": mode,
        },
        "make_env": make_env,
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "num_iters": num_iters,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "initial_batch_size": initial_batch_size,
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "num_eval_trajectories": num_eval_trajectories,
    }
