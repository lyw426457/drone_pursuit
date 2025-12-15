# experiments/run_maddpg.py
import os
import sys
import time

import numpy as np
import torch
import yaml

# 方便导入：将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.multi_drone_env import MultiDronePursuitEnv
from policy.maddpg import MADDPG
from utils.utils import save_model


def load_config():
    """从 experiments/config.yaml 读取超参数，并补默认值。"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ===== 基础超参 =====
    cfg.setdefault("learning_rate", 5e-4)
    cfg.setdefault("num_episodes", 50000)
    cfg.setdefault("batch_size", 256)
    cfg.setdefault("gamma", 0.95)
    cfg.setdefault("tau", 0.01)

    # ===== Replay Buffer & 训练节奏 =====
    cfg.setdefault("buffer_capacity", 500000)
    cfg.setdefault("warmup_steps", 20000)
    cfg.setdefault("update_every", 50)
    cfg.setdefault("update_iters", 10)

    # ===== 环境基础参数 =====
    cfg.setdefault("initial_distance", 250.0)
    cfg.setdefault("pos_limit", 300.0)
    cfg.setdefault("max_steps", 800)
    cfg.setdefault("catch_threshold", 1.0)
    cfg.setdefault("pursuer_max_speed", 6.0)
    cfg.setdefault("evader_max_speed", 2.0)
    cfg.setdefault("detection_radius", 200.0)

    # ===== 三阶段参数 =====
    cfg.setdefault("takeoff_altitude", 30.0)
    cfg.setdefault("takeoff_forward_speed", 4.0)
    cfg.setdefault("takeoff_climb_rate", 2.0)
    cfg.setdefault("max_takeoff_steps", 200)
    cfg.setdefault("approach_speed_scale", 3.0)

    return cfg


def main():
    # ① 读取配置
    cfg = load_config()

    lr = cfg["learning_rate"]
    num_episodes = cfg["num_episodes"]
    batch_size = cfg["batch_size"]
    gamma = cfg["gamma"]
    tau = cfg["tau"]

    buffer_capacity = cfg["buffer_capacity"]
    warmup_steps = cfg["warmup_steps"]
    update_every = cfg["update_every"]
    update_iters = cfg["update_iters"]

    initial_distance = cfg["initial_distance"]
    pos_limit = cfg["pos_limit"]
    max_steps = cfg["max_steps"]
    catch_threshold = cfg["catch_threshold"]
    pursuer_max_speed = cfg["pursuer_max_speed"]
    evader_max_speed = cfg["evader_max_speed"]
    detection_radius = cfg["detection_radius"]

    takeoff_altitude = cfg["takeoff_altitude"]
    takeoff_forward_speed = cfg["takeoff_forward_speed"]
    takeoff_climb_rate = cfg["takeoff_climb_rate"]
    max_takeoff_steps = cfg["max_takeoff_steps"]
    approach_speed_scale = cfg["approach_speed_scale"]

    # ② 创建环境（带三阶段参数）
    env = MultiDronePursuitEnv(
        dt=0.1,
        max_steps=max_steps,
        pos_limit=pos_limit,
        catch_threshold=catch_threshold,
        pursuer_max_speed=pursuer_max_speed,
        evader_max_speed=evader_max_speed,
        initial_distance=initial_distance,
        detection_radius=detection_radius,
        takeoff_altitude=takeoff_altitude,
        takeoff_forward_speed=takeoff_forward_speed,
        takeoff_climb_rate=takeoff_climb_rate,
        approach_speed_scale=approach_speed_scale,
        max_takeoff_steps=max_takeoff_steps,
    )

    num_agents = 2
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Using device: {device}")
    print(
        f"[Config] lr={lr}, num_episodes={num_episodes}, batch_size={batch_size}, "
        f"gamma={gamma}, tau={tau}, buffer_capacity={buffer_capacity}, warmup_steps={warmup_steps}, "
        f"update_every={update_every}, update_iters={update_iters}"
    )
    print(
        f"[Env] initial_distance={initial_distance}, pos_limit={pos_limit}, max_steps={max_steps}, "
        f"catch_threshold={catch_threshold}, pursuer_max_speed={pursuer_max_speed}, "
        f"evader_max_speed={evader_max_speed}, detection_radius={detection_radius}, "
        f"takeoff_altitude={takeoff_altitude}, takeoff_forward_speed={takeoff_forward_speed}, "
        f"takeoff_climb_rate={takeoff_climb_rate}, max_takeoff_steps={max_takeoff_steps}, "
        f"approach_speed_scale={approach_speed_scale}"
    )

    # ③ 初始化 MADDPG（训练模式，eval_mode=False）
    maddpg = MADDPG(
        num_agents=num_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=gamma,
        tau=tau,
        actor_lr=lr,
        critic_lr=lr,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        device=device,
        eval_mode=False,
    )

    max_steps_env = env.max_steps
    total_steps = 0  # 全局 step 计数（跨 episode）

    # === 创建唯一目录（基于当前时间戳）保证每次训练模型单独保存 ===
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    project_root = os.path.dirname(os.path.dirname(__file__))
    logs_dir = os.path.join(project_root, "logs", "models")
    os.makedirs(logs_dir, exist_ok=True)
    model_run_dir = os.path.join(logs_dir, f"run_{timestamp}")
    os.makedirs(model_run_dir, exist_ok=True)
    print(f"[Main] Model checkpoints will be saved to: {model_run_dir}")

    # ④ 训练循环前：初始化“历史最佳成绩”
    best_score = -float("inf")  # 越大越好
    best_ep = -1

    for ep in range(num_episodes):
        obs_n, info = env.reset()
        ep_rewards = [0.0 for _ in range(num_agents)]
        done = False
        step = 0
        dist_sum = 0.0
        last_phase = info.get("phase", "approach")

        while not done and step < max_steps_env:
            step += 1
            total_steps += 1

            # warmup：只收集经验
            if total_steps < warmup_steps:
                actions_n = []
                for _ in range(num_agents):
                    a = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
                    actions_n.append(a)
            else:
                actions_n = maddpg.select_actions(obs_n, noise_std=0.1)

            next_obs_n, rew_n, done, info = env.step(actions_n)
            dist_sum += info["dist"]

            phase_str = info.get("phase", "pursuit")

            # ===== 将 phase 写入 TensorBoard（0/1/2）=====
            phase_code = {"approach": 0, "takeoff": 1, "pursuit": 2}[phase_str]
            # 确保 MADDPG 内部有 self.total_steps 计数器
            maddpg.writer.add_scalar("env/phase", phase_code, maddpg.total_steps)

            # ===== 只在 phase 发生变化时打印一行 =====
            if phase_str != last_phase:
                print(
                    f"[Episode {ep+1:05d} | Step {step:03d}] "
                    f"PHASE CHANGE → {phase_str}, dist={info['dist']:.2f}"
                )
                last_phase = phase_str

            # 只在 pursuit 阶段收集经验 / 训练
            if phase_str == "pursuit":
                maddpg.store_transition(obs_n, actions_n, rew_n, next_obs_n, done)

            obs_n = next_obs_n
            for i in range(num_agents):
                ep_rewards[i] += rew_n[i]

            # 按频率更新
            if total_steps >= warmup_steps and (total_steps % update_every == 0):
                for _ in range(update_iters):
                    maddpg.train_step()

        # ===== Episode 级别打印 =====
        print(
            f"[Episode {ep+1:05d}] "
            f"reward_pursuer={ep_rewards[0]:.2f}, reward_evader={ep_rewards[1]:.2f}, "
            f"caught={info.get('caught', False)}, last_dist={info['dist']:.3f}"
        )

        # ===== 写入 TensorBoard =====
        writer = maddpg.writer
        mean_dist = dist_sum / max(step, 1)
        caught_flag = 1.0 if info.get("caught", False) else 0.0

        # 综合评分：既考虑平均距离，又考虑是否抓到
        score = -mean_dist + 10.0 * caught_flag

        if writer is not None:
            writer.add_scalar("episode/reward_pursuer", ep_rewards[0], ep)
            writer.add_scalar("episode/reward_evader", ep_rewards[1], ep)
            writer.add_scalar("episode/length", step, ep)
            writer.add_scalar("episode/mean_dist", mean_dist, ep)
            writer.add_scalar("episode/caught", caught_flag, ep)
            writer.add_scalar("episode/score", score, ep)
            writer.add_scalar("debug/replay_buffer_size", len(maddpg.replay_buffer), ep)
            writer.add_scalar("debug/total_env_steps", total_steps, ep)

        # ===== 只保存“最好的模型” =====
        if score > best_score:
            best_score = score
            best_ep = ep + 1

            for i, actor in enumerate(maddpg.actors):
                save_path = os.path.join(model_run_dir, f"best_actor_agent_{i}.pth")
                save_model(actor, save_path)

            print(
                f"[Best] New best model at episode {ep+1}, "
                f"score={score:.3f}, mean_dist={mean_dist:.3f}, caught={bool(caught_flag)}"
            )

    print(f"[Main] Training finished. Best episode = {best_ep}, best_score = {best_score:.3f}")


if __name__ == "__main__":
    main()
