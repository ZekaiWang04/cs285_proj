import os
import time
from typing import Optional
from matplotlib import pyplot as plt
import yaml
from cs285 import envs

from cs285.agents.ode_agent import ODEAgent
from cs285.agents.model_based_agent import ModelBasedAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse

from cs285.envs import register_envs

register_envs()


def run_training_loop(
    config: dict, agent_name: str, logger: Logger, args: argparse.Namespace
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our MPC implementation only supports continuous action spaces."

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 2

    mb_agent=utils.RandomPolicy(env=env)
    actor_agent = mb_agent

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    total_envsteps = 0

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        if itr == 0:
            # TODO(student): collect at least config["initial_batch_size"] transitions with a random policy
            # HINT: Use `utils.RandomPolicy` and `utils.sample_trajectories`
            trajs, envsteps_this_batch = utils.sample_trajectories(env=env, 
                                                                   policy=utils.RandomPolicy(env), 
                                                                   min_timesteps_per_batch=config["initial_batch_size"],
                                                                   max_length=ep_len)
        else:
            # TODO(student): collect at least config["batch_size"] transitions with our `actor_agent`
            trajs, envsteps_this_batch = utils.sample_trajectories(env=env, 
                                                                   policy=actor_agent, 
                                                                   min_timesteps_per_batch=config["batch_size"],
                                                                   max_length=ep_len)

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        for traj in trajs:
            replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )


        # train agent
        print("Training agent...")
        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return: {np.mean(returns)}")

        if len(returns) > 1:
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    actor_agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    itr,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    config, agent_name = make_config(args.config_file)
    logger = make_logger(config)


    run_training_loop(config, agent_name, logger, args)


if __name__ == "__main__":
    main()
