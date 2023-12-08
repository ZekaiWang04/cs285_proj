from typing import Optional
import numpy as np
import gym
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, Dopri5
import optax
from cs285.envs.dt_sampler import BaseSampler

class NeuralODE(eqx.Module):
    _str_to_activation = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "leaky_relu": jax.nn.leaky_relu,
        "sigmoid": jax.nn.sigmoid,
        "selu": jax.nn.selu,
        "softplus": jax.nn.softplus,
        "identity": lambda x: x,
    }
    mlp: eqx.nn.MLP
    def __init__(
            self,
            hidden_size,
            num_layers,
            ob_dim,
            ac_dim,
            key,
            activation="relu",
            output_activation="identity",
        ):
        super().__init__()
        activation = self._str_to_activation[activation]
        output_activation = self._str_to_activation[output_activation]
        # hidden_size is an integer
        self.mlp = eqx.nn.MLP(in_size=ob_dim+ac_dim,
                              out_size=ob_dim,
                              width_size=hidden_size,
                              depth=num_layers,
                              activation=activation,
                              final_activation=output_activation,
                              key=key)

    @eqx.filter_jit
    def __call__(self, t, y, args):
        # args is a dictionary that contains times and actions
        times = args["times"] # (ep_len,)
        actions = args["actions"] # (ep_len, ac_dim)
        idx = jnp.searchsorted(times, t, side="right") - 1
        action = actions[idx] # (ac_dim,)
        # althoug I believe this should also work for batched
        return self.mlp(jnp.concatenate((y, action), axis=-1))
    
class ODEAgent():
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
        train_timestep: float,
        train_discount: float,
        mpc_horizon_steps: int,
        mpc_discount: float,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        mpc_dt_sampler: BaseSampler,
        mpc_timestep: float,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        # super().__init__()
        self.env = env
        self.train_timestep = train_timestep
        assert 0 < train_discount <= 1
        self.train_discount = train_discount
        self.mpc_horizon_steps = mpc_horizon_steps # in terms of timesteps
        assert 0 < mpc_discount <= 1
        self.mpc_discount = mpc_discount
        self.mpc_strategy = mpc_strategy
        self.mpc_timestep = mpc_timestep
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
        self.ode_functions = [NeuralODE(
            hidden_size=hidden_size,
            num_layers=num_layers,
            ob_dim=self.ob_dim,
            ac_dim=self.ac_dim,
            activation=activation,
            output_activation=output_activation,
            key = keys[n]
            ) for n in range(ensemble_size)]
        self.optims = [optax.adamw(lr) for _ in range(ensemble_size)]
        self.optim_states = [self.optims[n].init(eqx.filter(self.ode_functions[n], eqx.is_array)) for n in range(self.ensemble_size)]

        self.solver = Dopri5()
    
    # I believe only jitting the top level function should work...
    # need testing/reading to support this "conjecture"
    @DeprecationWarning
    def update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        """
        Update self.dynamics_models[i] using the given trajectory

        Args:
            i: index of the dynamics model to update
            obs: (ep_len, ob_dim)
            acs: (ep_len, ac_dim)
            times: (ep_len)
        """
        # TODO: add discount, train_length
        # Note: the discount will mess with the loss, so if we want to 
        # compare "training" effect with different discount, we can't 
        # really do that

        # TODO: for some reason, this function is an order of magnitude
        # slower than the batched_update function below. For now I have
        # deperacated this function in favor of the one below.
        ep_len = obs.shape[0]
        assert obs.shape == (ep_len, self.ob_dim)
        assert acs.shape == (ep_len, self.ac_dim)
        assert times.shape == (ep_len,)

        discount_array = self.train_discount ** jnp.arange(ep_len)[..., jnp.newaxis]

        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def loss_grad(ode_func):
            sol = diffeqsolve(
                diffrax.ODETerm(ode_func), 
                self.solver, 
                t0=times[0], 
                t1=times[-1],
                dt0=self.train_timestep,
                y0 = obs[0, :],
                args={"times": times, "actions": acs},
                saveat=diffrax.SaveAt(ts=times)
            )
            assert sol.ys.shape == obs.shape == (ep_len, self.ob_dim)
            return jnp.mean(discount_array * (sol.ys - obs) ** 2) # do we want a  "discount"-like trick

        @eqx.filter_jit
        def make_step(ode_func, optim, opt_state):
            loss, grad = loss_grad(ode_func)
            updates, opt_state = optim.update(grad, opt_state, ode_func)
            ode_func = eqx.apply_updates(ode_func, updates)
            return loss, ode_func, opt_state
        
        ode_func, optim, opt_state = self.ode_functions[i], self.optims[i], self.optim_states[i]
        loss, ode_func, opt_state = make_step(ode_func, optim, opt_state)
        self.ode_functions[i], self.optim_states[i] = ode_func, opt_state
        return loss.item()

    def batched_update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        batch_size, ep_len = times.shape[0], times.shape[1]
        assert times.shape == (batch_size, ep_len)
        assert obs.shape == (batch_size, ep_len, self.ob_dim)
        assert acs.shape == (batch_size, ep_len, self.ac_dim)

        discount_array = self.train_discount ** jnp.arange(ep_len)[..., jnp.newaxis]
        ode_func, optim, opt_state = self.ode_functions[i], self.optims[i], self.optim_states[i]

        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def get_batchified_loss(ode_func, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
            def get_single_loss(ob: jnp.ndarray, ac: jnp.ndarray, time: jnp.ndarray):
                assert ob.shape == (ep_len, self.ob_dim)
                assert ac.shape == (ep_len, self.ac_dim)
                assert time.shape == (ep_len,)
                sol = diffeqsolve(
                    terms=diffrax.ODETerm(ode_func),
                    solver=self.solver,
                    t0=time[0],
                    t1=time[-1],
                    dt0=self.train_timestep,
                    y0=ob[0],
                    args={"times": time, "actions": ac},
                    saveat=diffrax.SaveAt(ts=time)
                )
                assert sol.ys.shape == ob.shape == (ep_len, self.ob_dim)
                return jnp.mean(discount_array * (sol.ys - ob) ** 2)
            losses = jax.vmap(get_single_loss)(obs, acs, times)
            return jnp.mean(losses)
        
        loss, grad = get_batchified_loss(ode_func, obs, acs, times)
        updates, opt_state = optim.update(grad, opt_state, ode_func)
        ode_func = eqx.apply_updates(ode_func, updates)
        self.ode_functions[i], self.optim_states[i] = ode_func, opt_state
        return loss.item()

    @eqx.filter_jit
    def evaluate_action_sequences(self, obs: jnp.ndarray, acs: jnp.ndarray, mpc_discount_arr: jnp.ndarray):
        dts = self.mpc_dt_sampler.get_dt(size=(self.mpc_horizon_steps,))
        times = jnp.cumsum(dts) # (self.mpc_horizon_steps, )

        def evaluate_single_sequence(ac):
            avg_rewards = jnp.zeros((self.ensemble_size,))
            for i in range(self.ensemble_size):
                ode_func = self.ode_functions[i]
                ode_out = diffeqsolve(
                    terms=diffrax.ODETerm(ode_func),
                    solver=self.solver,
                    t0=times[0],
                    t1=times[-1],
                    dt0=self.mpc_timestep,
                    y0=obs,
                    args={"times": times, "actions": ac},
                    saveat=diffrax.SaveAt(ts=times)
                )
                rewards, _ = self.env.get_reward_jnp(ode_out.ys, ac)
                avg_rewards.at[i].set(jnp.mean(rewards * mpc_discount_arr))
            return jnp.mean(avg_rewards)
        
        avg_rewards = jax.vmap(evaluate_single_sequence)(acs) # (seqs,)
        assert avg_rewards.shape == (self.mpc_num_action_sequences,)
        return avg_rewards

    @eqx.filter_jit
    def get_action(self, obs: jnp.ndarray, key: jax.random.PRNGKey):
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
        mpc_discount_arr = self.mpc_discount ** jnp.arange(self.mpc_horizon_steps)

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences, mpc_discount_arr)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = jnp.argmax(rewards)
            return action_sequences[best_index, 0, :]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                if i == 0:
                    elite_mean = np.mean(action_sequences, axis=0)
                    elite_std = np.std(action_sequences, axis=0)
                assert elite_mean.shape == (self.mpc_horizon_steps, self.ac_dim)
                assert elite_std.shape == (self.mpc_horizon_steps, self.ac_dim)
                acs_key, key = jax.random.split(key)
                action_sequences = elite_mean + elite_std * jax.random.normal(key=acs_key, shape=(self.mpc_num_action_sequences, self.mpc_horizon_steps, self.ac_dim))
                # action_sequences = jnp.clip(action_sequences, self.env.action_space.low, self.env.action_space.high)
                rewards = self.evaluate_action_sequences(obs, action_sequences, mpc_discount_arr)
                top_j_idx = jnp.argsort(rewards)[-self.cem_num_elites:]
                top_j_acs = action_sequences[top_j_idx]
                elite_mean = (1 - self.cem_alpha) * elite_mean + self.cem_alpha * jnp.mean(top_j_acs, axis=0)
                elite_std = (1 - self.cem_alpha) * elite_std + self.cem_alpha * jnp.std(top_j_acs, axis=0)
            return jnp.clip(elite_mean[0], self.env.action_space.low, self.env.action_space.high)
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")