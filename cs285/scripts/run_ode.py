import os
import time
from typing import Optional
from matplotlib import pyplot as plt
from cs285.agents.utils import save_leaves, load_leaves
from cs285.agents.ode_agent import ODEAgent
from cs285.infrastructure.replay_buffer import ReplayBufferTrajectories

import os
import time

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

import argparse


def run_training_loop_ode(
    config: dict, agent_name: str, logger: Logger, args: argparse.Namespace, log_dir: str
):
    checkpoint_dir= os.path.join(log_dir, "checkpoint")
    assert agent_name == "ode"
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
    mb_agent = ODEAgent(env, **config["agent_kwargs"])
    replay_buffer = ReplayBufferTrajectories(capacity=config["replay_buffer_capacity"])
    actor_agent = mb_agent

    total_envsteps = 0

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        data_key, key = jax.random.split(key)
        if itr == 0:
            ntraj = config["initial_trajs"]
            trajs, envsteps_this_batch = utils.sample_n_trajectories(
                env=env,
                policy=utils.RandomPolicy(env=env),
                ntraj=ntraj,
                key=data_key,
                max_length=ep_len,
            )
        else:
            ntraj = config["trajs"]
            trajs, envsteps_this_batch = utils.sample_n_trajectories(
                env=env, 
                policy=actor_agent, 
                ntraj=ntraj,
                key=data_key,
                max_length=ep_len
            )

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        replay_buffer.add_rollouts(paths=trajs)

        # train agent
        print("Training agent...")
        all_losses = []
        for _ in tqdm.trange(
            config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            step_losses = []
            for i in range(mb_agent.ensemble_size):
                sample_key, key = jax.random.split(key)
                traj = replay_buffer.sample_rollouts(batch_size=config["train_batch_size"], key=sample_key)
                obs = utils.split_arr(np.array(traj["observations"]), length=config["train_ep_len"], stride=config["train_stride"]) # (batch_size, num_splitted, train_ep_len, dims)
                acs = utils.split_arr(np.array(traj["actions"]), length=config["train_ep_len"], stride=config["train_stride"]) # (batch_size, num_splitted, train_ep_len, dims)
                dts = utils.split_arr(np.array(traj["dts"])[..., np.newaxis], length=config["train_ep_len"], stride=config["train_stride"]).squeeze(-1) # (batch_size, num_splitted, train_ep_len)
                batch_size, num_splitted, train_ep_len, ob_dim = obs.shape
                ac_dim = acs.shape[-1]
                obs = jnp.array(obs).reshape(batch_size * num_splitted, train_ep_len, ob_dim)
                acs = jnp.array(acs).reshape(batch_size * num_splitted, train_ep_len, ac_dim)
                times = jnp.cumsum(dts, axis=-1).reshape(batch_size * num_splitted, train_ep_len)
                loss = mb_agent.batched_update(
                    i=i,
                    obs=obs, 
                    acs=acs, 
                    times=times
                )
                step_losses.append(loss)
            all_losses.append(np.mean(step_losses))

        # save the current model
        save_leaves(mb_agent, checkpoint_dir)

        """
        # this block will cause some weird error
        # on iteration 0, plot the full learning curve
        if itr == 0:
            plt.plot(all_losses)
            plt.title("Iteration 0: Dynamics Model Training Loss")
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.savefig(os.path.join(logger._log_dir, "itr_0_loss_curve.png"))
        """

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