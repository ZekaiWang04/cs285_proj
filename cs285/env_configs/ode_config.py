import torch.nn as nn
from cs285.infrastructure import pytorch_util as ptu
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import gym
import torch
from typing import Optional, Sequence


def mpc_config(
    env_name: str,
    exp_name: str,
    learning_rate: float = 1e-3,
    ensemble_size: int = 3,
    mpc_horizon: int = 10,
    mpc_strategy: str = "random",
    mpc_num_action_sequences: int = 1000,
    cem_num_iters: Optional[int] = None,
    cem_num_elites: Optional[int] = None,
    cem_alpha: Optional[float] = None,
    initial_batch_size: int = 20000,  # number of transitions to collect with random policy at the start
    batch_size: int = 8000,  # number of transitions to collect per per iteration thereafter
    train_batch_size: int = 512,  # number of transitions to train each dynamics model per iteration
    num_iters: int = 20,
    replay_buffer_capacity: int = 1000000,
    num_agent_train_steps_per_iter: int = 20,
    num_eval_trajectories: int = 10,
    hidden_dims: Sequence[int] = [128, 128, 128],
    timestep: float = 0.005,
    activation: str = "relu",
    output_activation: str = "identity"
):
    # hardcoded for this assignment
    if env_name == "pendulum-cs285-v0":
        ep_len = 200

    def make_optimizer(params: nn.ParameterList):
        return torch.optim.Adam(params, lr=learning_rate)

    def make_env(render: bool = False):
        return RecordEpisodeStatistics(
            gym.make(env_name, render_mode="single_rgb_array" if render else None),
        )

    log_string = f"{env_name}_{exp_name}_hiddendims{hidden_dims}_mpc{mpc_strategy}_horizon{mpc_horizon}_actionseq{mpc_num_action_sequences}"
    if mpc_strategy == "cem":
        log_string += f"_cem_iters{cem_num_iters}"

    return {
        "agent_kwargs": {
            "make_optimizer": make_optimizer,
            "ensemble_size": ensemble_size,
            "mpc_horizon": mpc_horizon,
            "mpc_strategy": mpc_strategy,
            "mpc_num_action_sequences": mpc_num_action_sequences,
            "cem_num_iters": cem_num_iters,
            "cem_num_elites": cem_num_elites,
            "cem_alpha": cem_alpha,
            "hidden_dims": hidden_dims,
            "timestep": timestep,
            "activation": activation,
            "output_activation": output_activation
        },
        "make_env": make_env,
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "num_iters": num_iters,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "initial_batch_size": initial_batch_size,
        "train_batch_size": train_batch_size,
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "num_eval_trajectories": num_eval_trajectories,
    }
