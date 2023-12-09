from typing import Optional
import gym
import jax
import jax.numpy as jnp
from jax.lax import scan
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, Dopri5, PIDController, ODETerm
import optax
from cs285.envs.dt_sampler import BaseSampler
from cs285.agents.ode_agent import ODEAgent_Vanilla

_str_to_activation = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "leaky_relu": jax.nn.leaky_relu,
    "sigmoid": jax.nn.sigmoid,
    "selu": jax.nn.selu,
    "softplus": jax.nn.softplus,
    "identity": lambda x: x,
}

class ODE_RNN(eqx.Module):
    rnn_cell: eqx.Module
    mlp_dynamics: eqx.nn.MLP
    mlp_ob_encoder: eqx.nn.MLP
    mlp_ob_decoder: eqx.nn.MLP
    dt0: float

    def __init__(
            self,
            rnn_type: str,
            mlp_dynamics_setup: dict,
            mlp_ob_encoder_setup: dict,
            mlp_ob_decoder_setup: dict,
            ob_dim: int,
            ac_dim: int,
            latent_dim: int,
            dt0: float,
            key: jax.random.PRNGKey,
        ):
        # each mlp_..._setup should contain hidden_size: int, num_layers: int, activation: str, output_activation: str
        dynamics_key, ob_ac_encoder_key, ob_decoder_key = jax.random.split(key, 3)
        self.mlp_dynamics = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=latent_dim,
            width_size=mlp_dynamics_setup["hidden_size"],
            depth=mlp_dynamics_setup["num_layers"],
            activation=_str_to_activation[mlp_dynamics_setup["activation"]],
            final_activation=_str_to_activation[mlp_dynamics_setup["output_activation"]],
            key=dynamics_key
        )
        self.mlp_ob_encoder = eqx.nn.MLP(
            in_size=ob_dim,
            out_size=latent_dim,
            width_size=mlp_ob_encoder_setup["hidden_size"],
            depth=mlp_ob_encoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ob_encoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ob_encoder_setup["output_activation"]],
            key=ob_ac_encoder_key
        )
        self.mlp_ob_decoder = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=ob_dim,
            width_size=mlp_ob_decoder_setup["hidden_size"],
            depth=mlp_ob_decoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ob_decoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ob_decoder_setup["output_activation"]],
            key=ob_decoder_key
        )
        self.dt0 = dt0
        if rnn_type == "gru":
            self.rnn_cell = eqx.nn.GRUCell(
                input_size=ac_dim,
                hidden_size=latent_dim,
            )
        elif rnn_type == "lstm":
            self.rnn_cell = eqx.nn.LSTMCell(
                input_size=ac_dim,
                hidden_size=latent_dim,
            )



    @eqx.filter_jit
    def __call__(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        # ob: (ob_dim,) at times[0]
        # acs: (ep_len, ac_dim)
        # times: (ep_len,)
        # returns obs_predicted: (ep_len, ob_dim)
        latent = self.mlp_ob_encoder(ob) # (latent_dim,)
        latent = self.rnn_cell(acs[0], latent) # (latent_dim,)

        def step(latent, ac_dt):
            # latent: (latent_dim,)
            # ac_dt: (ac_dim + 1,)
            ac, dt = ac_dt[:-1], ac_dt[-1]
            ode_out = diffeqsolve(
                terms=ODETerm(lambda t, y, args: self.mlp_dynamics(y)),
                solver=Dopri5(),
                t0=0.0,
                t1=dt,
                dt0=self.dt0,
                y0=latent,
                stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
                saveat=diffrax.SaveAt(ts=[dt])
            )
            latent = ode_out.ys[0]
            latent = self.rnn_cell(ac, latent)
            return latent, latent
        
        dts = jnp.diff(times)[..., jnp.newaxis] # (ep_len-1, 1)
        _, latents = scan(step, latent, jnp.concatenate([acs[1:], dts], axis=-1))
        latents = jnp.concatenate([latent[jnp.newaxis, ...], latents], axis=0) # (ep_len, latent_dim)
        obs_predicted = jax.vmap(self.mlp_ob_decoder)(latents) # (ep_len, ob_dim)
        return obs_predicted


class ODE_RNN_Agent(ODEAgent_Vanilla):
    # https://arxiv.org/pdf/1907.03907.pdf
    env: gym.Env
    train_timestep: float
    train_discount: float
    mpc_horizon_steps: int
    mpc_discount: float
    mpc_strategy: str
    mpc_num_action_sequences: int
    mpc_dt_sampler: BaseSampler
    mpc_timestep: float
    cem_num_iters: int
    cem_num_elites: int
    cem_alpha: float
    ac_dim: int
    ob_dim: int
    ensemble_size: int
    ode_functions: list
    optims: list
    optim_states: list
    solver: Dopri5
    latent_dim: int

    def __init__(
        self,
        env: gym.Env,
        key: jax.random.PRNGKey,
        rnn_type: str,
        latent_dim: int,
        mlp_dynamics_setup: dict,
        mlp_ob_encoder_setup: dict,
        mlp_ob_decoder_setup: dict,
        odernn_dt0: float,
        optimizer_name: str,
        optimizer_kwargs: dict,
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
        super().__init__(
            env=env,
            key=key,
            mlp_dynamics_setup=mlp_dynamics_setup,
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            ensemble_size=ensemble_size,
            train_timestep=train_timestep,
            train_discount=train_discount,
            mpc_horizon_steps=mpc_horizon_steps,
            mpc_discount=mpc_discount,
            mpc_strategy=mpc_strategy,
            mpc_num_action_sequences=mpc_num_action_sequences,
            mpc_dt_sampler=mpc_dt_sampler,
            mpc_timestep=mpc_timestep,
            cem_num_iters=cem_num_iters,
            cem_num_elites=cem_num_elites,
            cem_alpha=cem_alpha,
        )
        self.latent_dim = latent_dim
        keys = jax.random.split(key, ensemble_size)
        self.ode_functions = [
            ODE_RNN(
                rnn_type=rnn_type,
                mlp_dynamics_setup=mlp_dynamics_setup,
                mlp_ob_encoder_setup=mlp_ob_encoder_setup,
                mlp_ob_decoder_setup=mlp_ob_decoder_setup,
                ob_dim=self.ob_dim,
                ac_dim=self.ac_dim,
                latent_dim=latent_dim,
                dt0=odernn_dt0,
                key=keys[n]
            ) for n in range(ensemble_size)
        ]
        optimizer_class = getattr(optax, optimizer_name)
        self.optims = [optimizer_class(**optimizer_kwargs) for _ in range(ensemble_size)]
        self.optim_states = [self.optims[n].init(eqx.filter(self.ode_functions[n], eqx.is_array)) for n in range(self.ensemble_size)]

    def batched_update(self, i: int, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        batch_size, ep_len = times.shape[0], times.shape[1]
        assert times.shape == (batch_size, ep_len)
        assert obs.shape == (batch_size, ep_len, self.ob_dim)
        assert acs.shape == (batch_size, ep_len, self.ac_dim)

        discount_array = self.train_discount ** jnp.arange(ep_len)[..., jnp.newaxis]
        ode_func, optim, opt_state = self.ode_functions[i], self.optims[i], self.optim_states[i]

        @eqx.filter_jit # might take too long to compile
        @eqx.filter_value_and_grad
        def get_batchified_loss(ode_func, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
            def get_single_loss(ob: jnp.ndarray, ac: jnp.ndarray, time: jnp.ndarray):
                assert ob.shape == (ep_len, self.ob_dim)
                assert ac.shape == (ep_len, self.ac_dim)
                assert time.shape == (ep_len,)
                ob_pred = ode_func(ob[0], ac, time)
                assert ob_pred.shape == (ep_len, self.ob_dim)
                return jnp.mean(discount_array * (ob_pred - ob) ** 2)
            losses = jax.vmap(get_single_loss)(obs, acs, times)
            return jnp.mean(losses)
        
        loss, grad = get_batchified_loss(ode_func, obs, acs, times)
        updates, opt_state = optim.update(grad, opt_state, ode_func)
        ode_func = eqx.apply_updates(ode_func, updates)
        self.ode_functions[i], self.optim_states[i] = ode_func, opt_state
        return loss.item()

    @eqx.filter_jit
    def evaluate_action_sequences(self, ob: jnp.ndarray, acs: jnp.ndarray, mpc_discount_arr: jnp.ndarray):
        # ob: (ob_dim,)
        # acs: (mpc_num_action_sequences, mpc_horizon_steps, ac_dim)
        # mpc_discount_arr: (mpc_horizon_steps,)
        dts = self.mpc_dt_sampler.get_dt(size=(self.mpc_horizon_steps,))
        times = jnp.cumsum(dts) # (self.mpc_horizon_steps, )

        def evaluate_single_sequence(ac):
            avg_rewards = jnp.zeros((self.ensemble_size,))
            for i in range(self.ensemble_size):
                ode_func = self.ode_functions[i]
                obs_predicted = ode_func(ob, ac, times)
                rewards, _ = self.env.get_reward_jnp(obs_predicted, ac)
                avg_rewards.at[i].set(jnp.mean(rewards * mpc_discount_arr))
            return jnp.mean(avg_rewards)
        
        avg_rewards = jax.vmap(evaluate_single_sequence)(acs) # (seqs,)
        assert avg_rewards.shape == (self.mpc_num_action_sequences,)
        return avg_rewards