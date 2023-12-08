# CS 285 Final Project
This is a tracking document for our final project

## TODOs
### Implementation
##Next TODO##Change ODE structure, for example try latent ode, possibly incorporating action (see the attached paper for two latent ode based approaches: ODE-RNN and VAE-based, also try vanilla MLP encoder-decoder)

Be careful of dt

Train with different, manually set "ep_len" wity numpy slideing window view

Jaxify everything and then we can ditch ptu and torch.

Examine why on the notebook training is lightning speed while on the script it takes so long ("solution": use jupyter notebooks to do experiments!!!)
### Experiments

1. Dyamic learning with ODE 

2. True dynamics MPC & CEM

3. Baseline

4. Chanve Env setting





## Project Structure
### Dynamic learning with ODE

### Getting action using MPC with ODE dynamics

### Putting things together: ODE Agent

### (If time permits) Dyna-style / MBPO style with ODE dynamics

### Baselines




## Log

Jax implementation done! New bottleneck is again at training

Checked baseline performance on simplest task (run ```python cs285/scripts/run.py -cfg experiments/pendulum_multi_iter.yaml``` at commit 832a59e5a184ecb48c566d6e8584a99e454d7e4f)

Incorporated discount in ODE training
see ```notebooks/test_ode_agent_jax.ipynb```

Batchify ODE training (i.e. batched SGD)
see ```notebooks/test_ode_agent_jax.ipynbw```

Implemented CEM

Implemented skeleton for ode_agent_true_dynamics

Give up on trying to figure out why for ode_agent.update(), cpu runs faster than gpu. Deperated it in favor of the faster ode_agent.batch_update().

Change the run script / configs to have different training setups.

Have a mpc_dt_sampler and ensemble over different dt distributions

Get the true dynamics in the ode_agent_true_dynamics.py

Use Baseline that incorporates $\Delta s = f_\theta (s, a) dt$ or $\Delta s = f_\theta (s, a, dt)$ (recall currently the vanilla MPC is $\Delta s = f_\theta (s, a)$, which is agnostic to $dt$). Also we might want to try to incorporate $t$ into $f_\theta(\cdot)$ but I don't believe this will improve performance.
## Possible References

Latent ODE
https://arxiv.org/pdf/1907.03907.pdf
Introduces RNN-ODE and the variational inference based Latent ODE

Augmented ODE
https://arxiv.org/pdf/1904.01681.pdf
Simple hack of augmenting the ode, prelude to linear encoder decoder to latent space

https://arxiv.org/pdf/1810.01367.pdf
Some even more complicated paper on latent ode for generative model, maybe we don't need this

ODE Original Paper

ODE PhD Thesis

Jax

The Continuous-time Model Based