from gym.envs.registration import register

def register_envs():
    register(
        id='pendulum-cs285-v0',
        entry_point='cs285.envs.pendulum:PendulumEnv',
        max_episode_steps=1000,
    )