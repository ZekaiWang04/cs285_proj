import os
import time
from typing import Optional
from matplotlib import pyplot as plt
from cs285.agents.model_based_agent import ModelBasedAgent
from cs285.infrastructure.replay_buffer import ReplayBufferTransitions

import os
import time

import gym
import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse


def run_training_loop_mpc(
    config: dict, agent_name: str, logger: Logger, args: argparse.Namespace
):
    assert agent_name == "mpc"
    # set random seeds
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

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

    # initialize agent
    mb_agent = ModelBasedAgent(env, **config["agent_kwargs"])
    replay_buffer = ReplayBufferTransitions(config["replay_buffer_capacity"])
    actor_agent = mb_agent

    total_envsteps = 0

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        data_key, key = jax.random.split(key)
        if itr == 0:
            trajs, envsteps_this_batch = utils.sample_trajectories(env=env, 
                                                                   policy=utils.RandomPolicy(env), 
                                                                   min_timesteps_per_batch=config["initial_batch_size"],
                                                                   max_length=ep_len,
                                                                   key=data_key)
        else:
            trajs, envsteps_this_batch = utils.sample_trajectories(env=env, 
                                                                   policy=actor_agent, 
                                                                   min_timesteps_per_batch=config["batch_size"],
                                                                   max_length=ep_len,
                                                                   key=data_key)

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
                dts=traj["dt"]
            )

        # update agent's statistics with the entire replay buffer
        mb_agent.update_statistics(
            obs=replay_buffer.observations[: len(replay_buffer)],
            acs=replay_buffer.actions[: len(replay_buffer)],
            next_obs=replay_buffer.next_observations[: len(replay_buffer)],
        )

        # train agent
        print("Training agent...")
        all_losses = []
        for _ in tqdm.trange(
            config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            step_losses = []
            for i in range(mb_agent.ensemble_size):
                batch = replay_buffer.sample(config["train_batch_size"])
                loss = mb_agent.batched_update(i, batch["observations"], batch["actions"], batch["next_observations"], batch["dts"])
                step_losses.append(loss)
            all_losses.append(np.mean(step_losses))

        # on iteration 0, plot the full learning curve
        if itr == 0:
            plt.plot(all_losses)
            plt.title("Iteration 0: Dynamics Model Training Loss")
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.savefig(os.path.join(logger._log_dir, "itr_0_loss_curve.png"))

        # log the average loss
        loss = np.mean(all_losses)
        logger.log_scalar(loss, "dynamics_loss", itr)

        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        sample_key, key = jax.random.split(key)
        trajs, _ = utils.sample_n_trajectories(
            eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            key=sample_key,
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
                data_key, key = jax.random.split(key)
                video_trajectories = utils.sample_n_trajectories(
                    env=render_env,
                    policy=actor_agent,
                    ntraj=args.num_render_trajectories,
                    max_length=ep_len,
                    key=data_key,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    itr,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )