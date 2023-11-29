from typing import Callable, Optional, Tuple, Sequence
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu
from torchdiffeq import odeint_adjoint as odeint

class NeuralODE(nn.Module):
    _str_to_activation = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(),
        "identity": nn.Identity(),
    }
    def __init__(self, hidden_dims, ob_dim, ac_dim, activation="relu", output_activation='identity'):
        activation = self._str_to_activation[activation]
        output_activation = self._str_to_activation[output_activation]
        layers = []
        hidden_dims = [ob_dim + ac_dim] + hidden_dims
        for n in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[n], hidden_dims[n+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], ob_dim))
        layers.append(output_activation)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.net = nn.Sequential(*layers)
    
    def update_action(self, action):
        self.register_buffer("ac", action)

    def forward(self, t, y):
        return self.net(torch.cat(y, self.ac))

    
class ODEAgent(nn.Module):
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
        self.register_buffer("timestep", timestep)

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
                NeuralODE(
                    hidden_dims,
                    self.ob_dim,
                    self.ac_dim,
                    activation,
                    output_activation
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

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
        predicted_obs = torch.zeros(next_obs.shape)
        steps = torch.floor(dt / self.timestep) + 1
        steps = steps.to(int)
        ode_func = self.ode_functions[i]
        for n in range(batch_size):
            self.ode_func.update_action(acs[n, :])
            t = torch.linspace(0, dt[n], steps[n])
            ode_out = odeint(ode_func, obs[n, :], t)
            predicted_obs[n, :] = ode_out[-1, :]
        loss = self.loss_fn(next_obs, predicted_obs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)
    
    def update_statistics(self, **args):
        pass

    @torch.no_grad()
    def evaluate_action_sequences(self, obs: np.ndarray, acs: np.ndarray):
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        t = torch.linspace(0, self.mpc_horizon * self.timestep, self.mpc_horizon + 1)
        reward_arr = np.zeros(self.mpc_num_action_sequences, self.ensemble_size)
        for n in range(self.mpc_num_action_sequences):
            for i in range(self.ensemble_size):
                ode_func = self.ode_functions[i]
                ode_func.update_action(ptu.from_numpy(acs[n, :]))
                ode_out = odeint(ode_func, obs, t) # (self.mpc_horizon + 1, ob_dim)
                assert ode_out.shape[0] == self.mpc_horizon + 1
                rewards = self.env.get_reward(ptu.to_numpy(ode_out), ptu.to_numpy(acs.repeat(self.mpc_horizon + 1, 1)))
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