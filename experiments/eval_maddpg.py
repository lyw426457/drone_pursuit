# experiments/eval_maddpg.py

import glob
import os
import sys

import numpy as np
import torch
import yaml

# 先把上一级目录加到 sys.path，方便导入环境和策略
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(curr_dir)  # drone_pursuit/
sys.path.append(project_root)

from environment.multi_drone_env import MultiDronePursuitEnv
from policy.maddpg import MADDPG
from utils.utils import load_model
from visualization.acmi_export import export_traj_to_acmi


def load_config():
    config_path = os.path.join(curr_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 默认值
    cfg.setdefault("max_steps", 500)
    cfg.setdefault("pos_limit", 200.0)
    cfg.setdefault("initial_distance", 100.0)
    cfg.setdefault("catch_threshold", 1.0)
    cfg.setdefault("pursuer_max_speed", 4.0)
    cfg.setdefault("evader_max_speed", 1.0)
    return cfg


def find_latest_run_dir(models_root: str) -> str:
    """
    在 logs/models/ 下找到最新的 run_XXXX 目录。
    按目录名排序（时间戳），取最后一个。
    """
    if not os.path.exists(models_root):
        raise FileNotFoundError(f"[Eval] models_root 不存在: {models_root}")

    candidates = [
        d
        for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d)) and d.startswith("run_")
    ]
    if not candidates:
        raise FileNotFoundError(f"[Eval] 在 {models_root} 下没有找到任何 run_ 开头的目录")

    candidates.sort()
    latest_run = candidates[-1]
    latest_run_dir = os.path.join(models_root, latest_run)
    print(f"[Eval] 找到最新的 run 目录: {latest_run_dir}")
    return latest_run_dir


def find_best_or_latest_actor_ckpt(run_dir: str, agent_index: int) -> str:
    """
    优先加载:
    1) best_actor_agent_{i}.pth
    如果不存在，再退回:
    2) actor_agent_{i}_ep*.pth 中 episode 最大的那个（兼容老版本）
    """
    # 1) 优先查 best_actor
    best_name = f"best_actor_agent_{agent_index}.pth"
    best_path = os.path.join(run_dir, best_name)
    if os.path.exists(best_path):
        print(f"[Eval] agent_{agent_index} 使用最优权重: {best_path}")
        return best_path

    # 2) 退回旧命名方式
    pattern = os.path.join(run_dir, f"actor_agent_{agent_index}_ep*.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"[Eval] 在 {run_dir} 中找不到 best_actor_agent_{agent_index}.pth "
            f"或 actor_agent_{agent_index}_ep*.pth"
        )

    def extract_ep(fname: str) -> int:
        base = os.path.basename(fname)
        parts = base.split("_")
        ep_part = [p for p in parts if p.startswith("ep")]
        if not ep_part:
            return 0
        ep_str = ep_part[0].replace("ep", "").replace(".pth", "")
        try:
            return int(ep_str)
        except ValueError:
            return 0

    files.sort(key=extract_ep)
    latest_file = files[-1]
    print(f"[Eval] agent_{agent_index} 使用（旧格式）权重: {latest_file}")
    return latest_file


def evaluate(num_eval_episodes: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 1. 确定 models 根目录 和 最新 run_ 目录 =====
    models_root = os.path.join(project_root, "logs", "models")
    latest_run_dir = find_latest_run_dir(models_root)

    # ===== 2. 读取 config 并初始化环境（与训练保持一致） =====
    cfg = load_config()
    env = MultiDronePursuitEnv(
        dt=0.1,
        max_steps=cfg["max_steps"],
        pos_limit=cfg["pos_limit"],
        catch_threshold=cfg["catch_threshold"],
        pursuer_max_speed=cfg["pursuer_max_speed"],
        evader_max_speed=cfg["evader_max_speed"],
        initial_distance=cfg["initial_distance"],
    )

    num_agents = 2
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # ===== 3. 初始化 MADDPG 模型（eval_mode=True，不创建 TensorBoard 日志） =====
    maddpg = MADDPG(
        num_agents=num_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        eval_mode=True,
    )

    # ===== 4. 自动加载每个 agent 的权重 =====
    model_paths = []
    for i in range(num_agents):
        ckpt_path = find_best_or_latest_actor_ckpt(latest_run_dir, i)
        load_model(maddpg.actors[i], ckpt_path)
        model_paths.append(ckpt_path)

    print(f"[Eval] 成功加载的模型列表: ")
    for p in model_paths:
        print("   ", p)

    # ===== 5. 运行若干回合评估（无噪声） =====
    for ep in range(num_eval_episodes):
        obs_n, _ = env.reset()
        done = False
        step = 0

        while not done and step < env.max_steps:
            step += 1
            actions_n = maddpg.select_actions(obs_n, noise_std=0.0)
            obs_n, rew_n, done, info = env.step(actions_n)

        print(
            f"[Eval Episode {ep+1}] "
            f"caught={info['caught']}, final_dist={info['dist']:.3f}"
        )

    # ===== 6. 渲染最后一次回合的轨迹 =====
    env.render()
    print("[Eval] 评估结束，已渲染轨迹。")

    # ===== 7. 导出 Tacview ACMI 轨迹 =====
    latest_run = os.path.basename(latest_run_dir)
    acmi_dir = os.path.join(project_root, "logs", "tacview")
    os.makedirs(acmi_dir, exist_ok=True)

    acmi_filename = f"{latest_run}_episode_{ep+1}.acmi"
    acmi_path = os.path.join(acmi_dir, acmi_filename)

    # export_traj_to_acmi(env.traj_p, env.traj_e, env.dt, acmi_path)
    export_traj_to_acmi(
    env.traj_p,
    env.traj_e,
    env.dt,
    acmi_path,
    base_altitude=1000.0,   # 统一抬高到 1000 m 上空
    xy_scale=5.0            # 可视化时把水平距离放大 5 倍
)
    print(f"[Eval] 已导出 ACMI: {acmi_path}")


if __name__ == "__main__":
    evaluate(num_eval_episodes=5)
