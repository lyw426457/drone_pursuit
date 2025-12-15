import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from .replay_buffer import ReplayBuffer


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = mlp(
            [obs_dim, *hidden_sizes, act_dim],
            activation=nn.ReLU,
            output_activation=nn.Tanh,  # 动作范围 [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, joint_obs_dim: int, joint_act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(
            [joint_obs_dim + joint_act_dim, *hidden_sizes, 1],
            activation=nn.ReLU,
            output_activation=nn.Identity,
        )

    def forward(self, joint_obs: torch.Tensor, joint_act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_obs, joint_act], dim=-1)
        q = self.net(x)
        return q


class MADDPG:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.95,
        tau: float = 0.01,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        buffer_capacity: int = 100000,
        batch_size: int = 128,
        device: str = "auto",
        eval_mode: bool = False,
    ):
        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[MADDPG] Using device: {self.device}")

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eval_mode = eval_mode

        self.total_steps = 0

        joint_obs_dim = num_agents * obs_dim
        joint_act_dim = num_agents * act_dim

        # 每个 agent 的 actor / critic
        self.actors: List[Actor] = []
        self.actors_target: List[Actor] = []
        self.critics: List[Critic] = []
        self.critics_target: List[Critic] = []
        self.actor_opts: List[optim.Optimizer] = []
        self.critic_opts: List[optim.Optimizer] = []

        for _ in range(num_agents):
            actor = Actor(obs_dim, act_dim).to(self.device)
            actor_t = Actor(obs_dim, act_dim).to(self.device)
            actor_t.load_state_dict(actor.state_dict())

            critic = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            critic_t = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            critic_t.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.actors_target.append(actor_t)
            self.critics.append(critic)
            self.critics_target.append(critic_t)

            self.actor_opts.append(optim.Adam(actor.parameters(), lr=actor_lr))
            self.critic_opts.append(optim.Adam(critic.parameters(), lr=critic_lr))

        # 经验回放池
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_agents=num_agents,
            capacity=buffer_capacity,
        )

        # TensorBoard：训练模式下才创建
        if not self.eval_mode:
            formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            logdir = os.path.join("logs", "runs", f"drone_pursuit_maddpg_{formatted_time}")
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=logdir)
            print(f"[MADDPG] TensorBoard log dir: {logdir}")
        else:
            self.writer = None  # eval 时不写日志

        # 训练步计数
        self.train_step_count = 0

    # ---------- 动作选择 ----------

    def select_actions(self, obs_n: List[np.ndarray], noise_std: float = 0.1) -> List[np.ndarray]:
        """
        obs_n: list of num_agents obs, each (obs_dim,)
        return: list of actions, each (act_dim,)
        """
        actions = []
        for i in range(self.num_agents):
            obs = torch.as_tensor(
                obs_n[i], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                a = self.actors[i](obs)  # [-1, 1]
            a = a.cpu().numpy()[0]
            if noise_std > 0:
                a = a + noise_std * np.random.randn(*a.shape)
            a = np.clip(a, -1.0, 1.0)
            actions.append(a.astype(np.float32))
        return actions

    # ---------- 存储经验 ----------

    def store_transition(self, obs_n, act_n, rew_n, next_obs_n, done: bool):
        # 统一记录环境步数（用于 TensorBoard）
        self.total_steps += 1

        self.replay_buffer.store(obs_n, act_n, rew_n, next_obs_n, done)

    # ---------- soft update ----------

    @torch.no_grad()
    def _soft_update(self, net, net_t):
        for p, p_t in zip(net.parameters(), net_t.parameters()):
            p_t.data.mul_(1 - self.tau)
            p_t.data.add_(self.tau * p.data)

    # ---------- 训练更新 ----------

    def train_step(self, log_interval: int = 100):
        """
        执行一次基于当前 replay buffer 的训练更新。
        - 当 buffer 中样本不足一个“安全值”时，直接 return。
        - 每次调用都会增加 self.train_step_count。
        - 支持打印 loss 和写入 TensorBoard。
        """
        # eval 模式下不应该被调用，但防御一下
        if self.eval_mode:
            return

        # 最小 buffer 容量：至少 5 个 batch，且不低于 5000
        min_buffer_size = max(self.batch_size * 5, 5000)
        if len(self.replay_buffer) < min_buffer_size:
            return

        batch = self.replay_buffer.sample_batch(self.batch_size)

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        # 切分成每个智能体的 obs / act
        obs_n = [
            obs[:, i * self.obs_dim : (i + 1) * self.obs_dim] for i in range(self.num_agents)
        ]
        next_obs_n = [
            next_obs[:, i * self.obs_dim : (i + 1) * self.obs_dim]
            for i in range(self.num_agents)
        ]
        acts_n = [
            acts[:, i * self.act_dim : (i + 1) * self.act_dim]
            for i in range(self.num_agents)
        ]

        critic_losses = []
        actor_losses = []

        for i in range(self.num_agents):
            # ===== critic 更新 =====
            with torch.no_grad():
                next_acts_n = []
                for j in range(self.num_agents):
                    a_next = self.actors_target[j](next_obs_n[j])
                    next_acts_n.append(a_next)
                next_acts = torch.cat(next_acts_n, dim=-1)  # (B, N*act_dim)

                target_q = self.critics_target[i](next_obs, next_acts).squeeze(-1)  # (B,)
                yi = rews[:, i] + self.gamma * (1.0 - done.squeeze(-1)) * target_q

            current_q = self.critics[i](obs, acts).squeeze(-1)

            critic_loss = nn.MSELoss()(current_q, yi.detach())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            # 梯度裁剪
            clip_grad_norm_(self.critics[i].parameters(), max_norm=0.5)
            self.critic_opts[i].step()

            # ===== actor 更新 =====
            curr_acts_n = []
            for j in range(self.num_agents):
                if j == i:
                    a_j = self.actors[j](obs_n[j])
                else:
                    # 其他 agent 的策略视为固定
                    a_j = self.actors[j](obs_n[j]).detach()
                curr_acts_n.append(a_j)
            curr_acts = torch.cat(curr_acts_n, dim=-1)

            actor_loss = -self.critics[i](obs, curr_acts).mean()
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actors[i].parameters(), max_norm=0.5)
            self.actor_opts[i].step()

            # soft update target 网络
            self._soft_update(self.actors[i], self.actors_target[i])
            self._soft_update(self.critics[i], self.critics_target[i])

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())

        # ===== 统计与日志 =====
        self.train_step_count += 1
        step = self.train_step_count

        agent_names = ["pursuer", "evader"]
        if self.writer is not None:
            for i in range(self.num_agents):
                agent_name = agent_names[i]
                self.writer.add_scalar(f"{agent_name}/critic_loss", critic_losses[i], step)
                self.writer.add_scalar(f"{agent_name}/actor_loss", actor_losses[i], step)

        if step % log_interval == 0:
            loss_str = " | ".join(
                f"{agent_names[i]}: critic={critic_losses[i]:.4f}, actor={actor_losses[i]:.4f}"
                for i in range(self.num_agents)
            )
            print(f"[train_step {step}] {loss_str}")
