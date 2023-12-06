# CS 285 Final Project
This is a tracking document for our final project

## TODOs
### Implementation
1. Incorporate discount in ODE training
2. Batchify ODE training
3. CEM
4. Debug why jax runs training faster on mac CPU than GPU
5. Add fully-jaxified implementation of everything, including from environment (should be easy: jax.jit and replacing np with jnp)
6. Change the run script / parsing to have different training setups (for example train with fixed length, discount, or other tricks)
7. Change ODE structure, for example try latent ode, possibly incorporating action
8. Use Baseline that incorporates $\Delta s = f_\theta (s, a) dt$ or $\Delta s = f_\theta (s, a, dt)$ (recall currently the vanilla MPC is $\Delta s = f_\theta (s, a)$, which is agnostic to $dt$). Also we might want to try to incorporate $t$ into $f_\theta(\cdot)$ but I don't believe this will improve performance.

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


