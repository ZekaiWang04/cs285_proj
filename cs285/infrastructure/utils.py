from collections import OrderedDict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import copy
import jax
import gym
import cv2
from typing import Dict, Tuple, List

from tqdm import trange

############################################
############################################


class RandomPolicy:
    def __init__(self, env: gym.Env):
        self.env = env

    def get_action(self, *_, **__):
        return self.env.action_space.sample()


def sample_trajectory(
    env: gym.Env, policy, max_length: int, key: jax.random.PRNGKey, render: bool = False,
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, dones, image_obs, dts = [], [], [], [], [], [], []
    steps = 0

    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="rgb_array")

            if isinstance(img, list):
                img = img[0]

            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        ac_key, key = jax.random.split(key)
        ac = policy.get_action(ob, key=ac_key)

        next_ob, rew, done, info = env.step(ac)
        dt = info["dt"]

        steps += 1
        # only record a "done" into the replay buffer if not truncated
        done_not_truncated = (
            done and steps <= max_length and not info.get("TimeLimit.truncated", False)
        )

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        dones.append(done_not_truncated)
        dts.append(dt)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if done or steps > max_length:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])

    env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "done": np.array(dones, dtype=np.float32),
        "episode_statistics": episode_statistics,
        "dt": np.array(dts, dtype=np.float32)
    }


def sample_trajectories(
    env: gym.Env,
    policy,
    min_timesteps_per_batch: int,
    max_length: int,
    key: jax.random.PRNGKey,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        sample_key, key = jax.random.split(key)
        traj = sample_trajectory(env, policy, max_length, sample_key, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy, ntraj: int, max_length: int, key: jax.random.PRNGKey, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    timesteps_this_bath = 0
    for i in trange(ntraj):
        # collect rollout
        sample_key, key = jax.random.split(key)
        traj = sample_trajectory(env, policy, max_length, sample_key, render)
        trajs.append(traj)

        # count steps
        timesteps_this_bath += get_traj_length(traj)
    return trajs, timesteps_this_bath


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    dts = np.concatenate([path["dt"] for path in paths])
    return observations, actions, rewards, next_observations, terminals,



def get_traj_length(traj):
    return len(traj["reward"])


def split_arr(arr: np.ndarray, length: int, stride: int=1):
    # arr (..., ep_len, dims)
    # returns (..., batch_size, length, dims)
    return sliding_window_view(arr, window_shape=length, axis=-2).swapaxes(-1,-2)[..., ::stride, :, :]