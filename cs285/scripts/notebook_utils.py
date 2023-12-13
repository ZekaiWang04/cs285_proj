from cs285.infrastructure.utils import sample_n_trajectories
from cs285.infrastructure import utils
import numpy as np
from tqdm import trange
import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import trange
import matplotlib.pyplot as plt

def train(agent, i, replay_buffer, train_config, key):
    optim = agent.optims[i]
    opt_state = agent.optim_states[i]
    discount_array = train_config["discount"] ** jnp.arange(train_config["ep_len"])
    neural_ode = agent.neural_odes[i]

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def get_loss_grad(neural_ode, obs, acs, times):
        obs_pred = neural_ode.batched_pred(ob=obs[:, 0, :], acs=acs, times=times)
        l2_losses = jnp.sum((obs - obs_pred) ** 2, axis=-1) # (batch_size, ep_len)
        weighed_mse = jnp.mean(discount_array * l2_losses)
        return weighed_mse

    def get_data(sample_key):
        traj = replay_buffer.sample_rollouts(batch_size=train_config["batch_size"], key=sample_key)
        obs = utils.split_arr(np.array(traj["observations"]), length=train_config["ep_len"], stride=train_config["stride"])
        acs = utils.split_arr(np.array(traj["actions"]), length=train_config["ep_len"], stride=train_config["stride"])
        dts = utils.split_arr(np.array(traj["dts"])[..., np.newaxis], length=train_config["ep_len"], stride=train_config["stride"]).squeeze(-1)
        batch_size, num_splitted, train_ep_len, ob_dim = obs.shape
        ac_dim = acs.shape[-1]
        obs = jnp.array(obs).reshape(batch_size * num_splitted, train_ep_len, ob_dim)
        acs = jnp.array(acs).reshape(batch_size * num_splitted, train_ep_len, ac_dim)
        times = jnp.cumsum(dts, axis=-1).reshape(batch_size * num_splitted, train_ep_len)
        return obs, acs, times

    losses = []
    for step in trange(train_config["steps"]):
        sample_key, key = jax.random.split(key)
        obs, acs, times = get_data(sample_key)
        loss, grad = get_loss_grad(neural_ode, obs, acs, times)
        updates, opt_state = optim.update(grad, opt_state, neural_ode)
        neural_ode = eqx.apply_updates(neural_ode, updates)
        losses.append(loss.item())

    plt.plot(np.arange(len(losses)), losses)
    agent.neural_odes[i] = neural_ode
    agent.optims[i] = optim
    agent.optim_states[i] = opt_state
    return agent, losses


def test(agent, ntraj, key, plot=False):
    trajs, _ = sample_n_trajectories(agent.env, agent, ntraj=ntraj, max_length=200, key=key)
    rewards = [t["episode_statistics"]["r"] for t in trajs]
    if plot:
        plt.hist(rewards, bins=20)
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
    return rewards, stats