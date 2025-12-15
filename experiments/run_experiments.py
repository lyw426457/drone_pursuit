# experiments/run_experiments.py
import os
import sys
import torch
import yaml  # 新增：读取 config.yaml

# 方便相对导入
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.drone_env import DroneEnv
from policy.agent import Agent
from policy.trainer import Trainer


def load_config():
    """从 experiments/config.yaml 读取超参数"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 给一些默认值，防止缺某个字段时报错
    cfg.setdefault("learning_rate", 1e-3)
    cfg.setdefault("num_episodes", 1000)
    cfg.setdefault("batch_size", 32)  # 单智能体暂时没用到，但预留
    return cfg


def main():
    # ① 读取配置
    cfg = load_config()
    lr = cfg["learning_rate"]
    num_episodes = cfg["num_episodes"]

    env = DroneEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ② 自动选择 GPU / CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Detected device: {device}")
    print(f"[Config] learning_rate={lr}, num_episodes={num_episodes}")

    # ③ 使用 config 中的学习率
    agent = Agent(obs_dim=obs_dim, act_dim=act_dim, lr=lr, device=device)

    # ④ 使用 config 中的训练轮数
    trainer = Trainer(env, agent, num_episodes=num_episodes)

    rewards = trainer.train()

    # 画一下最后的轨迹
    env.render()


if __name__ == "__main__":
    main()
