from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        mpc_discount: float=1.0,
        mpc_dt: float=0.05,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
        mode: str="vanilla" # vanilla, dt_in, mul_dt
    ): 
        self.mode = mode
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        assert 0 < mpc_discount <= 1
        self.mpc_discount = mpc_discount
        self.mpc_dt = mpc_dt
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

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
        if self.mode == "vanilla":
            self.dynamics_models = nn.ModuleList(
                [
                    make_dynamics_model(
                        self.ob_dim,
                        self.ac_dim,
                    )
                    for _ in range(ensemble_size)
                ]
            )
        elif self.mode == "dt_in":
            self.dynamics_models = nn.ModuleList(
                [
                    make_dynamics_model(
                        self.ob_dim,
                        self.ac_dim + 1, # hack, this +1 is where dt goes in, hence its name
                    )
                    for _ in range(ensemble_size)
                ]
            )
        elif self.mode == "mul_dt":
            self.dynamics_models = nn.ModuleList(
                [
                    make_dynamics_model(
                        self.ob_dim,
                        self.ac_dim,
                    )
                    for _ in range(ensemble_size)
                ]
            )
        else:
            raise Exception
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, dts :np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
            dts: (batch_size,)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        dts = ptu.from_numpy(dts[..., np.newaxis])
        eps = 1e-5
        obs_delta = next_obs - obs
        obs_delta_normalized = (obs_delta - self.obs_delta_mean) / (self.obs_delta_std + eps)
        obs_acs = torch.cat((obs, acs), dim=1)
        obs_acs_normalized = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + eps)
        if self.mode == "vanilla":
            obs_delta_predicted = self.dynamics_models[i](obs_acs_normalized)
        elif self.mode == "dt_in":
            obs_delta_predicted = self.dynamics_models[i](torch.concat((obs_acs_normalized, dts), dim=1))
        elif self.mode == "mul_dt":
            obs_delta_predicted = dts * self.dynamics_models[i](obs_acs_normalized)
        else:
            raise Exception
        loss = self.loss_fn(obs_delta_normalized, obs_delta_predicted)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        obs_acs = torch.cat((obs, acs), dim=1)
        obs_delta = next_obs - obs
        self.obs_acs_mean = torch.mean(obs_acs, dim=0)
        self.obs_acs_std = torch.std(obs_acs, dim=0)
        self.obs_delta_mean = torch.mean(obs_delta, dim=0)
        self.obs_delta_std = torch.std(obs_delta, dim=0)



    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray, dts: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            dts: (batch_size,)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        dts = ptu.from_numpy(dts[..., np.newaxis])
        eps = 1e-5
        obs_acs = torch.cat((obs, acs), dim=1)
        obs_acs_normalized = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + eps)
        if self.mode == "vanilla":
            obs_delta_normalized = self.dynamics_models[i](obs_acs_normalized)
        elif self.mode == "dt_in":
            obs_delta_normalized = self.dynamics_models[i](torch.concat((obs_acs_normalized, dts), dim=1))
        elif self.mode == "mul_dt":
            obs_delta_normalized = dts * self.dynamics_models[i](obs_acs_normalized)
        else:
            raise Exception
        obs_delta = obs_delta_normalized * (self.obs_delta_std + eps) + self.obs_delta_mean
        pred_next_obs = obs + obs_delta
        return ptu.to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray, dts: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
            dts: (mpc_num_action_sequences, horizon)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        discount = 1

        for n in range(action_sequences.shape[1]):
            acs = action_sequences[:, n, :]
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            next_obs = np.array([self.get_dynamics_predictions(i, obs[i, :, :], acs, dts[:, n]) for i in range(self.ensemble_size)])
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            next_obs_reshaped = next_obs.reshape(self.ensemble_size * self.mpc_num_action_sequences, self.ob_dim)
            acs_reshaped = acs.repeat(self.ensemble_size).reshape(self.ensemble_size * self.mpc_num_action_sequences, self.ac_dim)
            rewards = self.env.get_reward(next_obs_reshaped, acs_reshaped)[0]
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards * discount
            discount *= self.mpc_discount

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray, key=None):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        dts = self.mpc_dt * np.ones(shape=(self.mpc_num_action_sequences, self.mpc_horizon))

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences, dts)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                if i == 0:
                    elite_mean = np.mean(action_sequences, axis=0)
                    elite_std = np.std(action_sequences, axis=0)
                assert elite_mean.shape == (self.mpc_horizon, self.ac_dim)
                assert elite_std.shape == (self.mpc_horizon, self.ac_dim)
                action_sequences = np.random.normal(elite_mean, elite_std, size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim))
                action_sequences = np.clip(action_sequences, self.env.action_space.low, self.env.action_space.high)
                assert action_sequences.shape == (self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim)
                rewards = self.evaluate_action_sequences(obs, action_sequences, dts)
                top_j_indices = np.argsort(rewards)[-self.cem_num_elites:]
                top_j_actions = action_sequences[top_j_indices]
                elite_mean = (1 - self.cem_alpha) * elite_mean + self.cem_alpha * np.mean(top_j_actions, axis=0)
                elite_std = (1 - self.cem_alpha) * elite_std + self.cem_alpha * np.std(top_j_actions, axis=0)
            return np.clip(elite_mean[0], self.env.action_space.low, self.env.action_space.high)

        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
