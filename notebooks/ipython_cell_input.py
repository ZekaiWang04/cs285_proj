import torch
import numpy as np
@torch.no_grad()
def evaluate_action_sequences(agent, obs: np.ndarray, acs: np.ndarray):
    obs = ptu.from_numpy(obs)
    acs_np = acs
    acs = ptu.from_numpy(acs)
    times = torch.linspace(0, (agent.mpc_horizon_steps - 1) * agent.mpc_timestep, agent.mpc_horizon_steps, device=ptu.device)
    reward_arr = np.zeros((agent.mpc_num_action_sequences, agent.ensemble_size))
    for n in range(agent.mpc_num_action_sequences):
        for i in range(agent.ensemble_size):
            ode_func = agent.ode_functions[i]
            ode_func.update_action(acs[n, :, :], times)
            ode_out = odeint(ode_func, obs, times) # (steps, ob_dim)
            rewards, _ = agent.env.get_reward(ptu.to_numpy(ode_out), acs_np[n, :, :])
            avg_reward = np.mean(rewards)
            reward_arr[n, i] = avg_reward
    return np.mean(reward_arr, axis=1)

@torch.no_grad()
def get_action(agent, obs: np.ndarray):
    """
    Choose the best action using model-predictive control.

    Args:
        obs: (ob_dim,)
    """
    # always start with uniformly random actions
    actions = np.random.uniform(
        agent.env.action_space.low,
        agent.env.action_space.high,
        size=(agent.mpc_num_action_sequences, agent.mpc_horizon_steps, agent.ac_dim),
    )

    if agent.mpc_strategy == "random":
        # evaluate each action sequence and return the best one
        rewards = evaluate_action_sequences(agent, obs, actions)
        assert rewards.shape == (agent.mpc_num_action_sequences,)
        best_index = np.argmax(rewards)
        return actions[best_index, 0, :]
    elif agent.mpc_strategy == "cem":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid MPC strategy '{agent.mpc_strategy}'")

get_action(mb_agent, ob)
