import jax
import jax.numpy as jnp
from jax.lax import scan
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, Dopri5, PIDController, ODETerm
from typing import Optional

_str_to_activation = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "leaky_relu": jax.nn.leaky_relu,
    "sigmoid": jax.nn.sigmoid,
    "selu": jax.nn.selu,
    "softplus": jax.nn.softplus,
    "identity": lambda x: x,
}



class Base_NeuralODE(eqx.Module):
    def __init__(self):
        pass
    
    @eqx.filter_jit
    def single_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        """
        Given the ob at times[0], an acs array, and times array
        Generate the predicted obs at times

        Args:
            ob: (ob_dim,)
            acs: (ep_len, ac_dim)
            times: (ep_len,)

        Returns:
            obs_pred: (ep_len, ob_dim)
        """
        pass

    @eqx.filter_jit
    def batched_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        """
        Args:
            ob: (batch_size, ob_dim)
            acs: (batch_size, ep_len, ac_dim)
            times: (batch_size, ep_len)

        Returns:
            obs_pred: (batch_size, ep_len, ob_dim)
        """
        return jax.vmap(self.single_pred, (0, 0, 0), 0)(ob, acs, times)
    



class NeuralODE_Vanilla(Base_NeuralODE):
    mlp: eqx.nn.MLP
    ode_dt0: float
    def __init__(
        self,
        mlp_dynamics_setup: dict,
        ob_dim: int,
        ac_dim: int,
        ode_dt0: float,
        key: jax.random.PRNGKey,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=ob_dim+ac_dim,
            out_size=ob_dim,
            width_size=mlp_dynamics_setup["hidden_size"],
            depth=mlp_dynamics_setup["num_layers"],
            activation=_str_to_activation[mlp_dynamics_setup["activation"]],
            final_activation=_str_to_activation[mlp_dynamics_setup["output_activation"]],
            key=key
        )
        self.ode_dt0 = ode_dt0

    @eqx.filter_jit
    def _ode_term(self, t, y, args):
        # args is a dictionary that contains times and actions
        times = args["times"] # (ep_len,)
        actions = args["actions"] # (ep_len, ac_dim)
        idx = jnp.searchsorted(times, t, side="right") - 1
        action = actions[idx] # (ac_dim,)
        # althoug I believe this should also work for batched
        return self.mlp(jnp.concatenate((y, action), axis=-1))
    
    @eqx.filter_jit
    def single_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        # ob: (ob_dim,)
        # acs: (ep_len, ac_dim)
        # times: (ep_len,)
        # returns obs_pred: (ep_len, ob_dim)
        sol = diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: self._ode_term(t, y, args)),
            solver=Dopri5(),
            t0=times[0],
            t1=times[-1],
            dt0=self.ode_dt0,
            y0=ob,
            args={"times": times, "actions": acs},
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-5),
            saveat=diffrax.SaveAt(ts=times)
        )
        obs_pred = sol.ys
        return obs_pred
    


class Pendulum_True_Dynamics(Base_NeuralODE):
    ode_dt0: float
    def __init__(
        self,         
        ob_dim: Optional[int]=None,
        ac_dim: Optional[int]=None,
        ode_dt0: Optional[float]=None,
        key: Optional[jax.random.PRNGKey]=None,
    ):
        super().__init__()
        self.ode_dt0 = ode_dt0
    
    @eqx.filter_jit
    def _angle_normalize(self, x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    
    @eqx.filter_jit
    def _dynamics(self, t, y, args):
        times = args["times"] # (ep_len)
        actions = args["actions"] # (ep_len, ac_dim)
        idx = jnp.searchsorted(times, t, side="right") - 1
        action = actions[idx] # (ac_dim,)
        
        cos_theta, sin_theta, thdot = y
        th = self._angle_normalize(jnp.arctan2(sin_theta, cos_theta))
        max_speed = 8
        max_torque = 2.0
        g = 10.0
        m = 1.0
        l = 1.0
        u = jnp.clip(action, -max_torque, max_torque)[0]
        newthdot = thdot + (3 * g / (2 * l) * sin_theta + 3.0 / (m * l**2) * u) * (times[idx+1] - times[idx])
        newthdot = jnp.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * (times[idx+1] - times[idx])
        return jnp.asarray([-jnp.sin(newth) * newthdot, jnp.cos(newth) * newthdot, 3 * g / (2 * l) * sin_theta + 3.0 / (m * l**2) * u])
    
    @eqx.filter_jit
    def single_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        # ob: (ob_dim,)
        # acs: (ep_len, ac_dim)
        # times: (ep_len,)
        # returns obs_pred: (ep_len, ob_dim)
        sol = diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: self._dynamics(t, y, args)),
            solver=Dopri5(),
            t0=times[0],
            t1=times[-1],
            dt0=self.ode_dt0,
            y0=ob,
            args={"times": times, "actions": acs},
            # stepsize_controller=PIDController(rtol=1e-3, atol=1e-5),
            saveat=diffrax.SaveAt(ts=times),
        )
        obs_pred = sol.ys
        return obs_pred
    



class NeuralODE_Augmented(Base_NeuralODE):
    # https://arxiv.org/pdf/1904.01681.pdf, with minor changes
    mlp: eqx.nn.MLP
    aug_init_learnable: bool
    aug_init: jnp.ndarray
    aug_dim: int
    ode_dt0: float

    def __init__(
        self,
        mlp_dynamics_setup: dict,
        ob_dim: int,
        ac_dim: int,
        aug_dim: int,
        key: jax.random.PRNGKey,
        ode_dt0: float,
        aug_init_learnable: bool=False
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=ob_dim+ac_dim + aug_dim,
            out_size=ob_dim + aug_dim,
            width_size=mlp_dynamics_setup["hidden_size"],
            depth=mlp_dynamics_setup["num_layers"],
            activation=_str_to_activation[mlp_dynamics_setup["activation"]],
            final_activation=_str_to_activation[mlp_dynamics_setup["output_activation"]],
            key=key
        )
        self.aug_init_learnable = aug_init_learnable
        self.aug_dim = aug_dim
        self.aug_init = jnp.zeros((aug_dim,)) # TODO: better initialization
        self.ode_dt0 = ode_dt0
    
    @eqx.filter_jit # maybe invalid
    def _get_aug_init(self):
        if self.aug_init_learnable:
            return self.aug_init
        else:
            return jnp.zeros((self.aug_dim,))

    @eqx.filter_jit
    def _dynamics(self, t, y, args):
        # here y has shape (ob_dim + aug_dim,)
        times = args["times"] # (ep_len,)
        actions = args["actions"] # (ep_len, ac_dim)
        idx = jnp.searchsorted(times, t, side="right") - 1
        action = actions[idx] # (ac_dim,)
        return self.mlp(jnp.concatenate((y, action), axis=-1))
    
    @eqx.filter_jit
    def single_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        # ob: (ob_dim,)
        # acs: (ep_len, ac_dim)
        # times: (ep_len,)
        # returns obs_pred: (ep_len, ob_dim)
        sol = diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: self._dynamics(t, y, args)),
            solver=Dopri5(),
            t0=times[0],
            t1=times[-1],
            dt0=self.ode_dt0,
            y0=jnp.concatenate([ob, self._get_aug_init()], axis=-1), # maybe invalid
            args={"times": times, "actions": acs},
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-5),
            saveat=diffrax.SaveAt(ts=times)
        )
        obs_pred = sol.ys[:, :-self.aug_dim]
        return obs_pred
    




class NeuralODE_Latent_MLP(Base_NeuralODE):
    # runs in latent space
    # this implementation encodes/decodes both obs and acs
    mlp_dynamics: eqx.nn.MLP
    mlp_ob_encoder: eqx.nn.MLP
    mlp_ob_decoder: eqx.nn.MLP
    mlp_ac_encoder: eqx.nn.MLP
    ode_dt0: float

    def __init__(
        self,
        mlp_dynamics_setup: dict,
        mlp_ob_encoder_setup: dict,
        mlp_ob_decoder_setup: dict,
        mlp_ac_encoder_setup: dict, # no need for ac decoder
        ob_dim: int,
        ac_dim: int,
        ob_latent_dim: int,
        ac_latent_dim: int,
        ode_dt0: float,
        key: jax.random.PRNGKey,
    ):
        # each mlp_..._setup should contain hidden_size: int, num_layers: int, activation: str, output_activation: str
        super().__init__()
        dynamics_key, ob_encoder_key, ob_decoder_key, ac_encoder_key = jax.random.split(key, 4)
        self.mlp_dynamics = eqx.nn.MLP(
            in_size=ob_latent_dim+ac_latent_dim,
            out_size=ob_latent_dim,
            width_size=mlp_dynamics_setup["hidden_size"],
            depth=mlp_dynamics_setup["num_layers"],
            activation=_str_to_activation[mlp_dynamics_setup["activation"]],
            final_activation=_str_to_activation[mlp_dynamics_setup["output_activation"]],
            key=dynamics_key
        )
        self.mlp_ob_encoder = eqx.nn.MLP(
            in_size=ob_dim,
            out_size=ob_latent_dim,
            width_size=mlp_ob_encoder_setup["hidden_size"],
            depth=mlp_ob_encoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ob_encoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ob_encoder_setup["output_activation"]],
            key=ob_encoder_key
        )
        self.mlp_ob_decoder = eqx.nn.MLP(
            in_size=ob_latent_dim,
            out_size=ob_dim,
            width_size=mlp_ob_decoder_setup["hidden_size"],
            depth=mlp_ob_decoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ob_decoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ob_decoder_setup["output_activation"]],
            key=ob_decoder_key
        )
        self.mlp_ac_encoder = eqx.nn.MLP(
            in_size=ac_dim,
            out_size=ac_latent_dim,
            width_size=mlp_ac_encoder_setup["hidden_size"],
            depth=mlp_ac_encoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ac_encoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ac_encoder_setup["output_activation"]],
            key=ac_encoder_key
        )
        self.ode_dt0 = ode_dt0
    
    @eqx.filter_jit
    def _dynamics(self, t, y, args):
        # here y has shape (ob_latent_dim,)
        times = args["times"] # (ep_len,)
        ac_latents = args["ac_latents"] # (ep_len, ac_latent_dim)
        idx = jnp.searchsorted(times, t, side="right") - 1
        ac_latent = ac_latents[idx] # (ac_latent_dim,)
        return self.mlp_dynamics(jnp.concatenate((y, ac_latent), axis=-1))
    
    @eqx.filter_jit
    def single_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        # ob: (ob_dim,)
        # acs: (ep_len, ac_dim)
        # times: (ep_len,)
        # returns obs_pred: (ep_len, ob_dim)
        ac_latents = jax.vmap(self.mlp_ac_encoder)(acs) # (ep_len, ac_latent_dim)
        ob_latent = self.mlp_ob_encoder(ob) # (ob_latent_dim)
        sol = diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: self._dynamics(t, y, args)),
            solver=Dopri5(),
            t0=times[0],
            t1=times[-1],
            dt0=self.ode_dt0,
            y0=ob_latent,
            args={"times": times, "ac_latents": ac_latents},
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-5),
            saveat=diffrax.SaveAt(ts=times)
        )
        obs_pred = jax.vmap(self.mlp_ob_decoder)(sol.ys)
        return obs_pred



class ODE_RNN(Base_NeuralODE):
    rnn_cell: eqx.Module
    mlp_dynamics: eqx.nn.MLP
    mlp_ob_encoder: eqx.nn.MLP
    mlp_ob_decoder: eqx.nn.MLP
    ode_dt0: float
    rnn_type: str
    latent_dim: int

    def __init__(
        self,
        rnn_type: str,
        mlp_dynamics_setup: dict,
        mlp_ob_encoder_setup: dict,
        mlp_ob_decoder_setup: dict,
        ob_dim: int,
        ac_dim: int,
        latent_dim: int,
        ode_dt0: float,
        key: jax.random.PRNGKey,
    ):
        super().__init__()
        # each mlp_..._setup should contain hidden_size: int, num_layers: int, activation: str, output_activation: str
        dynamics_key, ob_ac_encoder_key, ob_decoder_key, rnn_key = jax.random.split(key, 4)
        self.mlp_ob_decoder = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=ob_dim,
            width_size=mlp_ob_decoder_setup["hidden_size"],
            depth=mlp_ob_decoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ob_decoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ob_decoder_setup["output_activation"]],
            key=ob_decoder_key
        )
        self.latent_dim = latent_dim
        self.ode_dt0 = ode_dt0
        self.rnn_type = rnn_type
        if rnn_type == "gru":
            self.rnn_cell = eqx.nn.GRUCell(
                input_size=ac_dim,
                hidden_size=latent_dim,
                key=rnn_key
            )
            total_hidden_dims = latent_dim
        elif rnn_type == "lstm":
            self.rnn_cell = eqx.nn.LSTMCell(
                input_size=ac_dim,
                hidden_size=latent_dim,
                key=rnn_key
            )
            total_hidden_dims = 2 * latent_dim

        self.mlp_ob_encoder = eqx.nn.MLP(
            in_size=ob_dim,
            out_size=total_hidden_dims,
            width_size=mlp_ob_encoder_setup["hidden_size"],
            depth=mlp_ob_encoder_setup["num_layers"],
            activation=_str_to_activation[mlp_ob_encoder_setup["activation"]],
            final_activation=_str_to_activation[mlp_ob_encoder_setup["output_activation"]],
            key=ob_ac_encoder_key
        )
        self.mlp_dynamics = eqx.nn.MLP(
            in_size=total_hidden_dims,
            out_size=total_hidden_dims,
            width_size=mlp_dynamics_setup["hidden_size"],
            depth=mlp_dynamics_setup["num_layers"],
            activation=_str_to_activation[mlp_dynamics_setup["activation"]],
            final_activation=_str_to_activation[mlp_dynamics_setup["output_activation"]],
            key=dynamics_key
        )


    @eqx.filter_jit
    def single_pred(self, ob: jnp.ndarray, acs: jnp.ndarray, times: jnp.ndarray):
        # ob: (ob_dim,) at times[0]
        # acs: (ep_len, ac_dim)
        # times: (ep_len,)
        # returns obs_predicted: (ep_len, ob_dim)

        # lstm: (h, c) is passed in to the cell as latents, h is the output
        def call_rnn(ac, latent):
            # ac: (ac_dim,)
            # latent: (total_hidden_dims,)
            # output: new_latent (total_hidden_dims,)
            if self.rnn_type == "gru":
                new_latent = self.rnn_cell(ac, latent)
            elif self.rnn_type == "lstm":
                latents = (latent[:self.latent_dim], latent[-self.latent_dim:])
                new_latents = self.rnn_cell(ac, latents)
                new_latent = jnp.concatenate(new_latents, axis=-1)
            else:
                raise NotImplementedError
            return new_latent
        
        latent = self.mlp_ob_encoder(ob) # (total_hidden_dim,)
        latent = call_rnn(acs[0], latent) # (total_hidden_dim,)
        def step(latent, ac_dt):
            # latent: (total_hidden_dim,)
            # ac_dt: (ac_dim + 1,)
            ac, dt = ac_dt[:-1], ac_dt[-1]
            ode_out = diffeqsolve(
                terms=ODETerm(lambda t, y, args: self.mlp_dynamics(y)),
                solver=Dopri5(),
                t0=0.0,
                t1=dt,
                dt0=self.ode_dt0,
                y0=latent,
                stepsize_controller=PIDController(rtol=1e-3, atol=1e-5),
                saveat=diffrax.SaveAt(ts=[dt])
            )
            latent = ode_out.ys[0]
            latent = call_rnn(ac, latent)
            return latent, latent
        
        dts = jnp.diff(times)[..., jnp.newaxis] # (ep_len-1, 1)
        _, latents = scan(step, latent, jnp.concatenate([acs[1:], dts], axis=-1))
        if self.rnn_type == "gru":
            latents = jnp.concatenate([latent[jnp.newaxis, ...], latents], axis=0) # (ep_len, latent_dim)
        elif self.rnn_type == "lstm":
            latents = jnp.concatenate([latent[jnp.newaxis, :self.latent_dim], latents[:, :self.latent_dim]], axis=0) # (ep_len, latent_dim)
        else:
            raise NotImplementedError
        obs_predicted = jax.vmap(self.mlp_ob_decoder)(latents) # (ep_len, ob_dim)
        return obs_predicted