from typing import Callable, Optional, Tuple, Sequence
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu
from torchdiffeq import odeint

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
        super().__init__()
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        activation = self._str_to_activation[activation]
        output_activation = self._str_to_activation[output_activation]
        layers = []
        hidden_dims = [ob_dim + ac_dim] + hidden_dims
        for n in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[n], hidden_dims[n+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], ob_dim))
        layers.append(output_activation)
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def update_action(self, actions: torch.Tensor, times: torch.Tensor):
        ep_len = actions.shape[0]
        assert actions.shape == (ep_len, self.ac_dim) and times.shape == (ep_len,)
        # times = times - times[0] # start with t=0
        # right now, do not assume t0 = 0
        self.register_buffer("times", times)
        self.register_buffer("actions", actions)

    def _get_action(self, t):
        idx = torch.searchsorted(self.times, t, right=True) - 1
        return self.actions[idx]

    def forward(self, t, y):
        ac = self._get_action(t)
        return self.net(torch.cat((y, ac), dim=-1))

    
class ODEAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        hidden_dims: Sequence[int],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon_steps: int,
        mpc_timestep: float,
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
        self.mpc_horizon_steps = mpc_horizon_steps # in terms of timesteps
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        self.mpc_timestep = mpc_timestep # when evaluating

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
                ).to(ptu.device)
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.ode_functions.parameters())
        self.loss_fn = nn.MSELoss()

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, times: np.ndarray):
        """
        Update self.dynamics_models[i] using the given trajectory

        Args:
            i: index of the dynamics model to update
            obs: (ep_len, ob_dim)
            acs: (ep_len, ac_dim)
            times: (ep_len)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        times = ptu.from_numpy(times)
        ode_func = self.ode_functions[i]
        ode_func.update_action(acs, times)
        ode_out = odeint(ode_func, obs[0, :], times) # t0 = times[0] in torchdiffeq
        # possible problem: the ode function is only "evaluating" on times
        # I am not sure whether there is an implicit dt or dt[i] = times[i+1] - times[i]
        # I know for diffrax in jax, there is a separate dt argument passed into odeint()
        assert ode_out.shape == obs.shape
        loss = self.loss_fn(ode_out, obs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)
    
    def update_statistics(self, **kwargs):
        pass

    @torch.no_grad()
    def evaluate_action_sequences(self, obs: np.ndarray, acs: np.ndarray):
        obs = ptu.from_numpy(obs) # (ob_dim)
        acs_np = acs
        acs = ptu.from_numpy(acs) # (N, steps, ac_dim)
        times = torch.linspace(0, (self.mpc_horizon_steps - 1) * self.mpc_timestep, self.mpc_horizon_steps, device=ptu.device)
        reward_arr = np.zeros((self.mpc_num_action_sequences, self.ensemble_size))
        for n in range(self.mpc_num_action_sequences):
            for i in range(self.ensemble_size):
                ode_func = self.ode_functions[i]
                ode_func.update_action(acs[n, :, :], times)
                ode_out = odeint(ode_func, obs, times) # (steps, ob_dim)
                rewards, _ = self.env.get_reward(ptu.to_numpy(ode_out), acs_np[n, :, :])
                avg_reward = np.mean(rewards)
                reward_arr[n, i] = avg_reward
        return np.mean(reward_arr, axis=1)
    # maybe I should manually implement batched Euler solver
    # to make inference faster

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
            size=(self.mpc_num_action_sequences, self.mpc_horizon_steps, self.ac_dim),
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