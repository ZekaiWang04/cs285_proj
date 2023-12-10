from typing import Optional
import gym
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, PIDController
import optax
from cs285.envs.dt_sampler import BaseSampler
from cs285.agents.nueral_ode import Base_NeuralODE, NeuralODE_Vanilla, Pendulum_True_Dynamics, NeuralODE_Augmented, NeuralODE_Latent_MLP, ODE_RNN
    
_neural_odes = {
    "vanilla": NeuralODE_Vanilla,
    "pendulum_true_dynamics": Pendulum_True_Dynamics,
    "augmented": NeuralODE_Augmented,
    "latent_mlp": NeuralODE_Latent_MLP,
    "ode_rnn": ODE_RNN
}

class ODEAgent(eqx.Module):
    env: gym.Env
    train_discount: float
    mpc_horizon_steps: int
    mpc_discount: float
    mpc_strategy: str
    mpc_num_action_sequences: int
    mpc_dt_sampler: BaseSampler
    cem_num_iters: int
    cem_num_elites: int
    cem_alpha: float
    ac_dim: int
    ob_dim: int
    ensemble_size: int
    neural_odes: list
    optims: list
    optim_states: list

    def __init__(
        self,
        env: gym.Env,
        key: jax.random.PRNGKey,
        neural_ode_name: str,
        neural_ode_kwargs: dict, # without key, ob_dim, ac_dim
        optimizer_name: str,
        optimizer_kwargs: dict,
        ensemble_size: int,
        train_discount: float,
        mpc_horizon_steps: int,
        mpc_discount: float,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        mpc_dt_sampler: BaseSampler,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        self.env = env
        assert 0 < train_discount <= 1
        self.train_discount = train_discount
        self.mpc_horizon_steps = mpc_horizon_steps # in terms of timesteps
        assert 0 < mpc_discount <= 1
        self.mpc_discount = mpc_discount
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha
        self.mpc_dt_sampler = mpc_dt_sampler # when evaluating

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
        keys = jax.random.split(key, ensemble_size)
        neural_ode_class = _neural_odes[neural_ode_name]
        self.neural_odes = [neural_ode_class(
            key=keys[n],
            ob_dim=self.ob_dim,
            ac_dim=self.ac_dim,
            **neural_ode_kwargs
            ) for n in range(ensemble_size)
        ]
        optimizer_class = getattr(optax, optimizer_name)
        self.optims = [optimizer_class(**optimizer_kwargs) for _ in range(ensemble_size)]
        self.optim_states = [self.optims[n].init(eqx.filter(self.neural_odes[n], eqx.is_array)) for n in range(self.ensemble_size)]

    def batched_update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        batch_size, ep_len = times.shape[0], times.shape[1]
        assert times.shape == (batch_size, ep_len)
        assert obs.shape == (batch_size, ep_len, self.ob_dim)
        assert acs.shape == (batch_size, ep_len, self.ac_dim)

        discount_array = self.train_discount ** jnp.arange(ep_len) # (ep_len,)
        neural_ode, optim, opt_state = self.neural_odes[i], self.optims[i], self.optim_states[i]

        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def get_loss(neural_ode, obs, acs, times):
            obs_pred = neural_ode.batched_pred(ob=obs[:, 0, :], acs=acs, times=times)
            l2_losses = jnp.sum((obs - obs_pred) ** 2, axis=-1) # (batch_size, ep_len)
            weighed_mse = jnp.mean(discount_array * l2_losses)
            return weighed_mse
        
        loss, grad = get_loss(neural_ode, obs, acs, times)
        updates, opt_state = optim.update(grad, opt_state, neural_ode)
        neural_ode = eqx.apply_updates(neural_ode, updates)
        self.neural_odes[i], self.optim_states[i] = neural_ode, opt_state
        return loss.item()

    @eqx.filter_jit
    def evaluate_action_sequences(self, ob: jnp.ndarray, acs: jnp.ndarray, mpc_discount_arr: jnp.ndarray):
        # ob: (ob_dim,)
        # acs: (mpc_num_action_sequences, mpc_horizon_steps, ac_dim)
        # mpc_discount_arr: (mpc_horizon_steps,)
        dts = self.mpc_dt_sampler.get_dt(size=(self.mpc_horizon_steps,))
        times = jnp.cumsum(dts) # (self.mpc_horizon_steps, )
        avg_rewards_ensembled = jnp.zeros((self.ensemble_size, self.mpc_num_action_sequences))
        for i in range(self.ensemble_size):
            neural_ode = self.neural_odes[i]
            obs_pred = neural_ode.batched_pred(
                ob=jnp.tile(ob, (self.mpc_num_action_sequences, 1)), 
                acs=acs, 
                times=jnp.tile(times, (self.mpc_num_action_sequences, 1))
            ) # TODO: is vmap and neural_ode.sigle_pred faster?
            rewards, _ = self.env.get_reward_jnp(obs_pred, acs)
            assert rewards.shape == (self.mpc_num_action_sequences, self.mpc_horizon_steps)
            avg_rewards_ensembled.at[i].set(jnp.mean(rewards * mpc_discount_arr, axis=-1))
        avg_rewards = jnp.mean(avg_rewards_ensembled, axis=0) # (mpc_num_action_sequences,)
        assert avg_rewards.shape == (self.mpc_num_action_sequences,)                                             
        return avg_rewards

    @eqx.filter_jit
    def get_action(self, ob: jnp.ndarray, key: jax.random.PRNGKey):
        """
        Choose the best action using model-predictive control.

        Args:
            ob: (ob_dim,)
        """
        # always start with uniformly random actions
        acs_key, key = jax.random.split(key)
        action_sequences = jax.random.uniform(
            key=acs_key,
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high,
            shape=(self.mpc_num_action_sequences, self.mpc_horizon_steps, self.ac_dim),
        )
        mpc_discount_arr = self.mpc_discount ** jnp.arange(self.mpc_horizon_steps)

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(ob, action_sequences, mpc_discount_arr)
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
                rewards = self.evaluate_action_sequences(ob, action_sequences, mpc_discount_arr)
                top_j_idx = jnp.argsort(rewards)[-self.cem_num_elites:]
                top_j_acs = action_sequences[top_j_idx]
                elite_mean = (1 - self.cem_alpha) * elite_mean + self.cem_alpha * jnp.mean(top_j_acs, axis=0)
                elite_std = (1 - self.cem_alpha) * elite_std + self.cem_alpha * jnp.std(top_j_acs, axis=0)
            return jnp.clip(elite_mean[0], self.env.action_space.low, self.env.action_space.high)
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")