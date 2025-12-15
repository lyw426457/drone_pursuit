# policy/trainer.py
import numpy as np


class Trainer:
    def __init__(self, env, agent, num_episodes: int = 50, max_steps: int | None = None):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps  # 如果为 None，用 env 的 max_steps

    def train(self):
        rewards = []
        for ep in range(self.num_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            step = 0
            done = False

            while not done:
                step += 1
                action = self.agent.act(obs)
                next_obs, reward, done, info = self.env.step(action)
                # 占位学习函数
                self.agent.learn_dummy(obs, action, reward, next_obs, done)

                obs = next_obs
                ep_reward += reward

                if self.max_steps is not None and step >= self.max_steps:
                    break

            rewards.append(ep_reward)
            print(f"[Episode {ep+1:03d}] total_reward = {ep_reward:.2f}, last_dist = {info['dist']:.3f}")

        return rewards
