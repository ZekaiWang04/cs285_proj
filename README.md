# CS 285 Final Project
This is a tracking document for our final project

## TODOs
### Implementation
1. ##Implementing## CEM
2. ##Next TODO## Debug why jax runs training faster on mac CPU than GPU
3. Add fully-jaxified implementation of everything, including from environment (should be easy: jax.jit and replacing np with jnp)
4. Change the run script / parsing to have different training setups (for example train with fixed length, discount, or other tricks)
5. Change ODE structure, for example try latent ode, possibly incorporating action
6. Use Baseline that incorporates $\Delta s = f_\theta (s, a) dt$ or $\Delta s = f_\theta (s, a, dt)$ (recall currently the vanilla MPC is $\Delta s = f_\theta (s, a)$, which is agnostic to $dt$). Also we might want to try to incorporate $t$ into $f_\theta(\cdot)$ but I don't believe this will improve performance.

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

### 12/5

Jax implementation done! New bottleneck is again at training

Checked baseline performance on simplest task (run ```python cs285/scripts/run.py -cfg experiments/pendulum_multi_iter.yaml``` at commit 832a59e5a184ecb48c566d6e8584a99e454d7e4f)

Incorporated discount in ODE training
see ```notebooks/test_ode_agent_jax.ipynb```

Batchify ODE training (i.e. batched SGD)
see ```notebooks/test_ode_agent_jax.ipynbw```

## Possible References

Latent ODE
https://arxiv.org/pdf/1907.03907.pdf

ODE Original Paper

ODE PhD Thesis

Jax

The Continuous-time Model Based