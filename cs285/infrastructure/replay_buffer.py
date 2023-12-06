from cs285.infrastructure.utils import *


class ReplayBufferTransitions:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dts = None
        self.dones = None

    def sample(self, batch_size):
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[rand_indices],
            "actions": self.actions[rand_indices],
            "rewards": self.rewards[rand_indices],
            "next_observations": self.next_observations[rand_indices],
            "dones": self.dones[rand_indices],
            "dts": self.dts[rand_indices]
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
        dt: np.ndarray
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                dt=dt
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)
        if isinstance(dt, float):
            dt = np.array(dt)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.dts = np.empty((self.max_size, *dt.shape), dtype=dt.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()
        assert dt.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.dts[self.size % self.max_size] = dt

        self.size += 1

    def batched_insert(
        self,
        /,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        dts: np.ndarray
        ):
        """
        Insert a batch of transitions into the replay buffer.
        """
        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observations.shape[1:]), dtype=observations.dtype
            )
            self.actions = np.empty(
                (self.max_size, *actions.shape[1:]), dtype=actions.dtype
            )
            self.rewards = np.empty(
                (self.max_size, *rewards.shape[1:]), dtype=rewards.dtype
            )
            self.next_observations = np.empty(
                (self.max_size, *next_observations.shape[1:]),
                dtype=next_observations.dtype,
            )
            self.dones = np.empty((self.max_size, *dones.shape[1:]), dtype=dones.dtype)
            self.dts = np.empty((self.max_size, *dts.shape[1:]), dtype=dts.dtype)

        assert observations.shape[1:] == self.observations.shape[1:]
        assert actions.shape[1:] == self.actions.shape[1:]
        assert rewards.shape[1:] == self.rewards.shape[1:]
        assert next_observations.shape[1:] == self.next_observations.shape[1:]
        assert dones.shape[1:] == self.dones.shape[1:]
        assert dts.shape[1:] == self.dts.shape[1:]

        indices = np.arange(self.size, self.size + observations.shape[0]) % self.max_size
        self.observations[indices] = observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_observations[indices] = next_observations
        self.dones[indices] = dones
        self.dts[indices] = dts

        self.size += observations.shape[0]


class ReplayBufferTrajectories():
    # There might be a bottleneck! Now I am using list, not array
    # This is because I want the flexibility to incorporate 
    # rollouts with different lengths
    # But I can also do something like having a cutoff_length
    # and a auxiliary value indicating how long each rollout is
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.dones = None
        self.dts = None

    def __len__(self):
        if self.obs is not None:
            return len(self.obs)
        else:
            return 0

    def add_rollouts(self, paths):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations = [path["observation"] for path in paths]
        actions = [path["action"] for path in paths]
        rewards = [path["reward"] for path in paths]
        next_observations = [path["next_observation"] for path in paths] 
        dones = [path["done"] for path in paths]
        dts = [path["dt"] for path in paths]

        if self.obs is None:
            self.obs = observations
            self.acs = actions
            self.rews = rewards
            self.next_obs = next_observations
            self.dones = dones
            self.dts = dts
        else:
            self.obs.extend(observations)
            self.acs.extend(actions)
            self.rews.extend(rewards)
            self.next_obs.extend(next_observations)
            self.dones.extend(dones)
            self.dts.extend(dts)

    def sample_rollout(self):
        # samples and returns a single rollout
        idx = self.rng.integers(low=0, high=len(self))
        return {
            "observations": self.obs[idx],
            "actions": self.acs[idx],
            "rewards": self.rews[idx],
            "next_observations": self.next_obs[idx],
            "dones": self.dones[idx],
            "dts": self.dts[idx]
        }
    
    def sample_rollouts(self, batch_size):
        indices = self.rng.integers(low=0, high=len(self), size=(batch_size,))
        observations, actions, rewards, next_observations, dones, dts = [], [], [], [], [], []
        for idx in indices:
            observations.append(self.obs[idx])
            actions.append(self.acs[idx])
            rewards.append(self.rews[idx])
            next_observations.append(self.next_obs[idx])
            dones.append(self.dones[idx])
            dts.append(self.dts[idx])
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "dones": dones,
            "dts": dts
        }



"""
    def sample_rollout(self, batch_size=1):
        # returns batch_size rollouts
        # Note: there is no "no-batch" option
        # so be careful with shapes when only
        # sampling one rollout
        idx = self.rng.integers(low=0, high=len(self), size=(batch_size,))
        return {
            "observations": self.obs[idx],
            "actions": self.acs[idx],
            "rewards": self.rews[idx],
            "next_observations": self.next_obs[idx],
            "dones": self.dones[idx],
            "dts": self.dts[idx]
        } # TODO: batchify!! or equivalently I don't batchify
    # and do weird tricks in the training script
    # need time to think about speed w/ jax
"""