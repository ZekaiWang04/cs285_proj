from typing import Callable, Optional, Tuple
import jax.numpy as jnp
import jax
import gym
from cs285.envs.dt_sampler import BaseSampler
import equinox as eqx
import optax


class ModelBasedAgent():
    _str_to_activation = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "leaky_relu": jax.nn.leaky_relu,
        "sigmoid": jax.nn.sigmoid,
        "selu": jax.nn.selu,
        "softplus": jax.nn.softplus,
        "identity": lambda x: x,
    }
    def __init__(
        self,
        env: gym.Env,
        key: jax.random.PRNGKey,
        hidden_size: int,
        num_layers: int,
        activation: str,
        output_activation: str,
        lr: float,
        ensemble_size: int,
        mpc_horizon_steps: int,
        mpc_discount: float,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        mpc_dt_sampler: BaseSampler,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
        mode: str="vanilla" # vanilla, dt_in, mul_dt
    ): 
        self.mode = mode
        super().__init__()
        self.env = env
        self.mpc_horizon_steps = mpc_horizon_steps
        self.mpc_strategy = mpc_strategy
        assert 0 < mpc_discount <= 1
        self.mpc_discount = mpc_discount
        self.mpc_dt_sampler = mpc_dt_sampler
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
        activation = self._str_to_activation[activation]
        output_activation = self._str_to_activation[output_activation]

        self.ensemble_size = ensemble_size
        keys = jax.random.split(key, ensemble_size)
        if self.mode == "vanilla":
            self.dynamics_models = [
                eqx.nn.MLP(
                    in_size=self.ob_dim + self.ac_dim,
                    out_size=self.ob_dim,
                    width_size=hidden_size,
                    depth=num_layers,
                    activation=activation,
                    final_activation=output_activation,
                    key=keys[n]
                )
                for n in range(ensemble_size)
            ]
        elif self.mode == "dt_in":
            self.dynamics_models = [
                eqx.nn.MLP(
                    in_size=self.ob_dim + self.ac_dim + 1, # dt
                    out_size=self.ob_dim,
                    width_size=hidden_size,
                    depth=num_layers,
                    activation=activation,
                    final_activation=output_activation,
                    key=keys[n]
                )
                for n in range(ensemble_size)
            ]
        elif self.mode == "mul_dt":
            self.dynamics_models = [
                eqx.nn.MLP(
                    in_size=self.ob_dim + self.ac_dim,
                    out_size=self.ob_dim,
                    width_size=hidden_size,
                    depth=num_layers,
                    activation=activation,
                    final_activation=output_activation,
                    key=keys[n]
                )
                for n in range(ensemble_size)
            ]
        else:
            raise Exception
        
        self.optims = [optax.adamw(lr) for _ in range(ensemble_size)]
        self.optim_states = [self.optims[n].init(eqx.filter(self.ode_functions[n], eqx.is_array)) for n in range(self.ensemble_size)]
        
        self.obs_acs_mean = jnp.zeros(self.ob_dim + self.ac_dim)
        self.obs_acs_std = jnp.ones(self.ob_dim + self.ac_dim)
        self.obs_delta_mean = jnp.zeros(self.ob_dim)
        self.obs_delta_std = jnp.ones(self.ob_dim)
        self.eps = 1e-5

    def batched_update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, next_obs: jnp.ndarray, dts :jnp.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
            dts: (batch_size,)
        """
        dts = dts[..., jnp.newaxis]
        obs_delta = next_obs - obs
        obs_delta_normalized = (obs_delta - self.obs_delta_mean) / (self.obs_delta_std + self.eps)
        obs_acs = jnp.concatenate((obs, acs), dim=1)
        obs_acs_normalized = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + self.eps)
        obs_acs_normalized_dts = jnp.concatenate((obs_acs_normalized, dts), axis=1)
        model, optim, opt_state = self.dynamics_models[i], self.optims[i], self.optim_states[i]

        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def get_batchified_loss(model, obs_acs_normalized_dts: jnp.ndarray, obs_delta_normalized: jnp.ndarray):
            def get_single_loss(ob_ac_normalized_dts: jnp.ndarray, ob_delta_normalized: jnp.array):
                assert ob_ac_normalized_dts.shape == (self.ob_dim + self.ac_dim + 1,)
                if self.mode == "vanilla":
                    ob_delta_predicted = model(ob_ac_normalized_dts[:-1])
                elif self.mode == "dt_in":
                    ob_delta_predicted = model(ob_ac_normalized_dts)
                elif self.mode == "mul_dt":
                    ob_delta_predicted = ob_ac_normalized_dts[-1] * model(ob_ac_normalized_dts[:-1])
                else:
                    raise Exception
                return jnp.sum((ob_delta_predicted, ob_delta_normalized) ** 2)
            losses = jax.vmap(get_single_loss, (0, 0), 0)(obs_acs_normalized_dts, obs_delta_normalized)
            return jnp.mean(losses)

        loss, grad = get_batchified_loss(model, obs_acs_normalized_dts, obs_delta_normalized)
        updates, opt_state = optim.update(grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        self.dynamics_models[i], self.optim_states[i] = model, opt_state
        return loss.item()

    def update_statistics(self, obs: jnp.ndarray, acs: jnp.ndarray, next_obs: jnp.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs_acs = jnp.concatenate((obs, acs), dim=1)
        obs_delta = next_obs - obs
        self.obs_acs_mean = jnp.mean(obs_acs, dim=0)
        self.obs_acs_std = jnp.std(obs_acs, dim=0)
        self.obs_delta_mean = jnp.mean(obs_delta, dim=0)
        self.obs_delta_std = jnp.std(obs_delta, dim=0)


    @eqx.filter_jit
    def get_dynamics_predictions(
        self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, dts: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            dts: (batch_size,)
        Returns: (batch_size, ob_dim)
        """
        dts = dts[..., jnp.newaxis]
        obs_acs = jnp.concatenate((obs, acs), dim=1)
        obs_acs_normalized = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + self.eps)
        obs_acs_normalized_dts = jnp.concatenate((obs_acs_normalized, dts), axis=1)
        model = self.dynamics_models[i]

        def forward(ob_ac_normalized_dts: jnp.ndarray):
            assert ob_ac_normalized_dts.shape == (self.ob_dim + self.ac_dim + 1,)
            if self.mode == "vanilla":
                ob_delta_predicted = model(ob_ac_normalized_dts[:-1])
            elif self.mode == "dt_in":
                ob_delta_predicted = model(ob_ac_normalized_dts)
            elif self.mode == "mul_dt":
                ob_delta_predicted = ob_ac_normalized_dts[-1] * model(ob_ac_normalized_dts[:-1])
            else:
                raise Exception
        obs_delta_normalized = jax.vmap(forward, 0, 0)(obs_acs_normalized_dts)
        obs_delta = obs_delta_normalized * (self.obs_delta_std + self.eps) + self.obs_delta_mean
        pred_next_obs = obs + obs_delta
        return pred_next_obs

    @eqx.filter_jit
    def evaluate_action_sequences(self, obs: jnp.ndarray, action_sequences: jnp.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        dts = self.mpc_discount.get_dt(size=(self.mpc_horizon_steps,))
        sum_of_rewards = jnp.zeros((self.ensemble_size, self.mpc_num_action_sequences))
        # We need to repeat our starting obs for each of the rollouts.
        obs = jnp.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))
        discount = 1

        for n in range(action_sequences.shape[1]):
            acs = action_sequences[:, n, :]
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            next_obs = jnp.array([self.get_dynamics_predictions(i, obs[i, :, :], acs, dts[n].repeat(self.mpc_num_action_sequences)) for i in range(self.ensemble_size)])
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            next_obs_reshaped = next_obs.reshape(self.ensemble_size * self.mpc_num_action_sequences, self.ob_dim)
            acs_reshaped = acs.repeat(self.ensemble_size).reshape(self.ensemble_size * self.mpc_num_action_sequences, self.ac_dim)
            rewards = self.env.get_reward_jnp(next_obs_reshaped, acs_reshaped)[0]
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards * discount
            discount *= self.mpc_discount

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    @eqx.filter_jit
    def get_action(self, obs: jnp.ndarray, key=None):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        acs_key, key = jax.random.split(key)
        action_sequences = jax.random.uniform(
            key=acs_key,
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high,
            shape=(self.mpc_num_action_sequences, self.mpc_horizon_steps, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = jnp.argmax(rewards)
            return action_sequences[best_index, 0, :]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                if i == 0:
                    elite_mean = jnp.mean(action_sequences, axis=0)
                    elite_std = jnp.std(action_sequences, axis=0)
                assert elite_mean.shape == (self.mpc_horizon_steps, self.ac_dim)
                assert elite_std.shape == (self.mpc_horizon_steps, self.ac_dim)
                acs_key, key = jax.random.split(key)
                action_sequences = elite_mean + elite_std * jax.random.normal(key=acs_key, shape=(self.mpc_num_action_sequences, self.mpc_horizon_steps, self.ac_dim))
                # action_sequences = jnp.clip(action_sequences, self.env.action_space.low, self.env.action_space.high)
                rewards = self.evaluate_action_sequences(obs, action_sequences)
                top_j_indices = jnp.argsort(rewards)[-self.cem_num_elites:]
                top_j_actions = action_sequences[top_j_indices]
                elite_mean = (1 - self.cem_alpha) * elite_mean + self.cem_alpha * jnp.mean(top_j_actions, axis=0)
                elite_std = (1 - self.cem_alpha) * elite_std + self.cem_alpha * jnp.std(top_j_actions, axis=0)
            return jnp.clip(elite_mean[0], self.env.action_space.low, self.env.action_space.high)

        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
