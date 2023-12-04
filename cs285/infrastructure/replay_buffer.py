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

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None
        self.dts = None

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
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
        terminals = [path["terminal"] for path in paths]
        dts = [path["dt"] for path in paths]

        if self.obs is None:
            self.obs = observations
            self.acs = actions
            self.rews = rewards
            self.next_obs = next_observations
            self.terminals = terminals
            self.dts = dts
        else:
            self.obs = self.obs.extend(observations)
            self.acs = self.acs.extend(actions)
            self.rews = self.rews.extend(rewards)
            self.next_obs = self.next_obs.extend(next_observations)
            self.terminals = self.terminals.extend(terminals)
            self.dts = self.dts.extend(dts)

    def sample_rollout(self):
        # returns a single rollout
        idx = self.rng.integers(low=0, high=len(self))
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
            "dts": self.dts[idx]
        }