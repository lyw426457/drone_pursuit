# policy/replay_buffer.py
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, num_agents: int, capacity: int = 100000):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.ptr = 0
        self.size = 0

        joint_obs_dim = obs_dim * num_agents
        joint_act_dim = act_dim * num_agents

        self.obs_buf = np.zeros((capacity, joint_obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, joint_obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, joint_act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, num_agents), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

    def store(
        self,
        obs_n,
        act_n,
        rew_n,
        next_obs_n,
        done: bool,
    ):
        # obs_n, next_obs_n: list of num_agents arrays (obs_dim,)
        # act_n: list of num_agents arrays (act_dim,)
        obs_concat = np.concatenate(obs_n, axis=0)
        next_obs_concat = np.concatenate(next_obs_n, axis=0)
        act_concat = np.concatenate(act_n, axis=0)
        rew_arr = np.asarray(rew_n, dtype=np.float32)

        idx = self.ptr
        self.obs_buf[idx] = obs_concat
        self.next_obs_buf[idx] = next_obs_concat
        self.act_buf[idx] = act_concat
        self.rew_buf[idx] = rew_arr
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch

    def __len__(self):
        return self.size
