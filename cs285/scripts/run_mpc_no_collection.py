import os
import time
from typing import Optional
from matplotlib import pyplot as plt
from cs285.agents.model_based_agent import ModelBasedAgent
from cs285.infrastructure.replay_buffer import ReplayBufferTransitions
from cs285.agents.utils import save_leaves, load_leaves
import os
import time

import gym
import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.scripts.scripting_utils import make_logger, make_config

import argparse
import pickle

def run_training_loop_mpc(
    config: dict, agent_name: str, logger: Logger, args: argparse.Namespace, log_dir: str
):
    checkpoint_dir= os.path.join(log_dir, "checkpoint")
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

    with open("./notebooks/reply_buffers/random_constant_0.05_replay_buffer", "rb") as f:
      replay_buffer = pickle.load(f)

    actor_agent = mb_agent

    total_envsteps = 0

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        data_key, key = jax.random.split(key)


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
                batch = replay_buffer.sample(config["batch_size"])
                loss = mb_agent.batched_update(i, batch["observations"], batch["actions"], batch["next_observations"], batch["dts"])
                step_losses.append(loss)
            all_losses.append(np.mean(step_losses))

        save_leaves(mb_agent, checkpoint_dir)

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
        rewards = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]
        plt.hist(rewards, bins=20)
        plt.show()
        mean, std, min, max = np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)
        print("mean", mean)
        print("std", std)
        print("min", min)
        print("max", max)
        stats = {
            "mean": mean,
            "std": std,
            "min": min,
            "max": max
        }
        print(f"Average eval return: {np.mean(rewards)}")
