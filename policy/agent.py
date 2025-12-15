# policy/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 输出用 tanh 压到 [-1, 1]
        return torch.tanh(self.net(obs))


class Agent:
    """
    极简版智能体：
    - act(): 给一个状态，返回一个 numpy 动作
    - learn_dummy(): 预留真正 RL 算法用
    """

    def __init__(self, obs_dim: int, act_dim: int, lr: float = 1e-3, device: str = "auto"):
        # -------- ① 自动选择设备 --------
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Agent] Using device: {self.device}")

        # -------- ② 把策略网络搬到 device --------
        self.policy = MLPPolicy(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """
        obs: numpy array, shape (obs_dim,)
        return: numpy array, shape (act_dim,)
        """
        self.policy.eval()

        # -------- ③ obs -> torch.Tensor on device --------
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            action_t = self.policy(obs_t)  # (1, act_dim) on device

        # -------- ④ tensor -> numpy (在 CPU 上) --------
        action = action_t.detach().cpu().numpy()[0]

        # 加一点探索噪声
        if noise_std > 0:
            action = action + noise_std * np.random.randn(*action.shape)

        # 连续动作 clip 到 [-1, 1]
        return np.clip(action, -1.0, 1.0)

    def learn_dummy(self, *args, **kwargs):
        """
        目前先不写具体的 RL 算法，这个函数只是个占位，
        确保 Trainer 调用不会报错。
        后续你可以在这里实现 DDPG/SAC 等。
        """
        pass
