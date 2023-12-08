from typing import Optional
import numpy as np
import gym
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve
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

class NeuralODE_Latent_MLP(eqx.Module):
    # runs in latent space
    # this implementation encodes/decodes obs but not acs
    # TODO: consider alternatives
    mlp_dynamics: eqx.nn.MLP
    mlp_encoder: eqx.nn.MLP
    mlp_decoder: eqx.nn.MLP

    def __init__(
            self,
            mlp_dynamics_setup: dict,
            mlp_encoder_setup: dict,
            mlp_decoder_setup: dict,
            ob_dim: int,
            ac_dim: int,
            latent_dim: int,
            key: jax.random.PRNGKey,
        ):
        # each mlp_..._setup should contain hidden_size: int, num_layers: int, activation: str, output_activation: str
        super().__init__()
        dynamics_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.mlp_dynamics = eqx.nn.MLP(
            in_size=latent_dim+ac_dim,
            out_size=latent_dim,
            width_size=mlp_dynamics_setup["hidden_size"],
            depth=mlp_dynamics_setup["num_layers"],
            activation=_str_to_activation[mlp_dynamics_setup["activation"]],
            final_activation=_str_to_activation[mlp_dynamics_setup["output_activation"]],
            key=dynamics_key
        )
        self.mlp_encoder = eqx.nn.MLP(
            in_size=ob_dim,
            out_size=latent_dim,
            width_size=mlp_encoder_setup["hidden_size"],
            depth=mlp_encoder_setup["num_layers"],
            activation=_str_to_activation[mlp_encoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_encoder_setup["output_activation"]],
            key=encoder_key
        )
        self.mlp_decoder = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=ob_dim,
            width_size=mlp_decoder_setup["hidden_size"],
            depth=mlp_decoder_setup["num_layers"],
            activation=_str_to_activation[mlp_decoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_decoder_setup["output_activation"]],
            key=decoder_key
        )
    
    @eqx.filter_jit
    def encode(self, ob: jnp.ndarray):
        # ob: (ob_dim,)
        return self.mlp_encoder(ob)

    @eqx.filter_jit
    def decode(self, latent: jnp.ndarray):
        # latent: (latent_dim,)
        return self.mlp_decoder(latent)
    
    @eqx.filter_jit
    def batched_decode(self, latents: jnp.ndarray):
        # latents: (batch_size, latent_dim)
        return jax.vmap(lambda latent: self.decode(latent))(latents)
    
    @eqx.filter_jit
    def __call__(self, t, y, args):
        # here y has shape (latent_dim,)
        times = args["times"] # (ep_len,)
        actions = args["actions"] # (ep_len, ac_dim)
        idx = jnp.searchsorted(times, t, side="right") - 1
        action = actions[idx] # (ac_dim,)
        return self.mlp((y, action), axis=-1)
    
class ODEAgent_Latent_MLP(ODEAgent_Vanilla):
    # ob --encoder--> z
    # dz/ dt = NN(latent, ac)
    # z --decoder--> ob
    # with encoder, NN, decoder all being MLPs
    # naturally generalizes the Augmented ODE
    def __init__(
        self,
        env: gym.Env,
        key: jax.random.PRNGKey,
        latent_dim: int,
        mlp_dynamics_setup: dict,
        mlp_encoder_setup: dict,
        mlp_decoder_setup: dict,
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
            NeuralODE_Latent_MLP(
                mlp_dynamics_setup=mlp_dynamics_setup,
                mlp_encoder_setup=mlp_encoder_setup,
                mlp_decoder_setup=mlp_decoder_setup,
                ob_dim=self.ob_dim,
                ac_dim=self.ac_dim,
                latent_dim=latent_dim,
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

        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def get_batchified_loss(ode_func, obs: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
            def get_single_loss(ob: jnp.ndarray, ac: jnp.ndarray, time: jnp.ndarray):
                assert ob.shape == (ep_len, self.ob_dim)
                assert ac.shape == (ep_len, self.ac_dim)
                assert time.shape == (ep_len,)
                z = ode_func.encode(ob) # (latent_dim,)
                sol = diffeqsolve(
                    terms=diffrax.ODETerm(ode_func),
                    solver=self.solver,
                    t0=time[0],
                    t1=time[-1],
                    dt0=self.train_timestep,
                    y0=z,
                    args={"times": time, "actions": ac},
                    saveat=diffrax.SaveAt(ts=time)
                )
                assert sol.ys.shape == (ep_len, self.latent_dim)
                ob_pred = ode_func.batched_decode(sol.ys)
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
                ode_out = diffeqsolve(
                    terms=diffrax.ODETerm(ode_func),
                    solver=self.solver,
                    t0=times[0],
                    t1=times[-1],
                    dt0=self.mpc_timestep,
                    y0=ode_func.encode(ob),
                    args={"times": times, "actions": ac},
                    saveat=diffrax.SaveAt(ts=times)
                )
                rewards, _ = self.env.get_reward_jnp(ode_func.batched_decode(ode_out.ys), ac)
                avg_rewards.at[i].set(jnp.mean(rewards * mpc_discount_arr))
            return jnp.mean(avg_rewards)
        
        avg_rewards = jax.vmap(evaluate_single_sequence)(acs) # (seqs,)
        assert avg_rewards.shape == (self.mpc_num_action_sequences,)
        return avg_rewards