from typing import Callable, Optional, Tuple, Sequence
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu
from torchdiffeq import odeint

class TrueDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        
    def update_action(self, action):
        self.register_buffer("ac", action)

    def forward(self, t, y):
        g = 10.0
        l = 1.0
        m = 1.0
        cos_theta = y[0]
        sin_theta = y[1]
        theta_dot = y[2]
        u = self.ac.squeeze()
        theta = torch.arctan2(sin_theta, cos_theta)
        newthdot = theta_dot + (3 * g / (2 * l) * torch.sin(theta) + 3.0 / (m * l**2) * u) * 0.05
        return torch.tensor([-sin_theta * newthdot, cos_theta * newthdot, newthdot - theta_dot], device=ptu.device)

class ODEAgent_cheat(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        hidden_dims: Sequence[int],
        timestep: float,
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
        activation: str = "relu",
        output_activation: str = "identity"
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        self.timestep = timestep

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.ode_functions = nn.ModuleList(
            [
                TrueDynamics().to(ptu.device) for _ in range(ensemble_size)
            ]
        )
        self.loss_fn = nn.MSELoss()

    @torch.no_grad() 
    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, dt: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
            dt: (batch_size)
        """
        batch_size = obs.shape[0]
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        dt = ptu.from_numpy(dt)
        predicted_obs = torch.zeros(next_obs.shape, device=ptu.device)
        steps = torch.floor(dt / self.timestep) + 1
        steps = steps.to(int)
        ode_func = self.ode_functions[i]
        for n in range(batch_size):
            ode_func.update_action(acs[n, :])
            t = torch.linspace(0, dt[n], steps[n], device=ptu.device)
            ode_out = odeint(ode_func, obs[n, :], t)
            predicted_obs[n, :] = ode_out[-1, :]
        loss = self.loss_fn(next_obs, predicted_obs)

        return ptu.to_numpy(loss)
    
    def update_statistics(self, **args):
        pass

    @torch.no_grad()
    def evaluate_action_sequences(self, obs: np.ndarray, acs: np.ndarray):
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        t = torch.linspace(0, self.mpc_horizon * self.timestep, self.mpc_horizon + 1, device=ptu.device)
        reward_arr = np.zeros((self.mpc_num_action_sequences, self.ensemble_size))
        for n in range(self.mpc_num_action_sequences):
            for i in range(self.ensemble_size):
                ode_func = self.ode_functions[i]
                ode_func.update_action(acs[n, :])
                ode_out = odeint(ode_func, obs, t) # (self.mpc_horizon + 1, ob_dim)
                assert ode_out.shape[0] == self.mpc_horizon + 1
                rewards = self.env.get_reward(ptu.to_numpy(ode_out), ptu.to_numpy(acs[n, :].repeat(self.mpc_horizon + 1, 1)))
                avg_reward = np.mean(rewards)
                reward_arr[n, i] = avg_reward
        return np.mean(reward_arr, axis=1)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        actions = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, actions)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return actions[best_index]
        elif self.mpc_strategy == "cem":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")