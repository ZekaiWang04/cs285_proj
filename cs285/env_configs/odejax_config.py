from cs285.infrastructure import pytorch_util as ptu
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import gym
import jax
from typing import Optional, Sequence
from cs285.envs.dt_sampler import BaseSampler, ConstantSampler, UniformSampler, ExponentialSampler


def odejax_config(
    env_name: str,
    exp_name: str,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    dt_sampler_name: str = "constant",
    dt_sampler_kwargs: dict={"dt": 0.05},
    learning_rate: float = 1e-3,
    ensemble_size: int = 3,
    mpc_horizon_steps: int = 100,
    mpc_timestep: float=0.005,
    mpc_strategy: str = "random",
    mpc_num_action_sequences: int = 1000,
    cem_num_iters: Optional[int] = None,
    cem_num_elites: Optional[int] = None,
    cem_alpha: Optional[float] = None,
    initial_trajs: int = 100,  # number of trajectories to collect with random policy at the start
    trajs: int = 8000,  # number of transitions to collect per per iteration thereafter
    train_batch_size: int = 512,  # number of transitions to train each dynamics model per iteration
    num_iters: int = 20,
    replay_buffer_capacity: int = 1000000,
    num_agent_train_steps_per_iter: int = 20,
    num_eval_trajectories: int = 10,
    hidden_size: int = 32,
    num_layers: int = 4,
    activation: str = "relu",
    output_activation: str = "identity",
    train_timestep: float = 0.005
):
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

    log_string = f"{env_name}_{exp_name}_hiddensize{hidden_size}_mpc{mpc_strategy}_horizon{mpc_horizon_steps}_actionseq{mpc_num_action_sequences}"
    if mpc_strategy == "cem":
        log_string += f"_cem_iters{cem_num_iters}"

    return {
        "agent_kwargs": {
            "key": key,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "ensemble_size": ensemble_size,
            "train_timestep": train_timestep,
            "mpc_horizon_steps": mpc_horizon_steps,
            "mpc_timestep": mpc_timestep,
            "mpc_strategy": mpc_strategy,
            "mpc_num_action_sequences": mpc_num_action_sequences,
            "cem_num_iters": cem_num_iters,
            "cem_num_elites": cem_num_elites,
            "cem_alpha": cem_alpha,
            "activation": activation,
            "output_activation": output_activation,
            "lr": learning_rate
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
    }
