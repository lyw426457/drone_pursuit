import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from .drone_model import DroneModel


class MultiDronePursuitEnv:
    """
    两架无人机的三维追逃环境（带三阶段场景逻辑）：
    - agent 0: pursuer（追捕者）
    - agent 1: evader（逃逸者）

    三阶段训练场景：
      1) "approach": 逃逸者从远处飞向基地，追捕者在地面待命（不训练）
      2) "takeoff": 逃逸者进入雷达圈，追捕者按固定翼方式滑跑 + 爬升（不训练）
      3) "pursuit": 起飞完成后，双方由 MADDPG 控制进行空中追逃（训练只在此阶段进行）

    状态：
      对每个 agent i，obs_i = [self_state(12), other_state(12)] -> 24 维
      其中 state = [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]

    动作：
      对每个 agent i，action_i = [vx_cmd, vy_cmd, vz_cmd] ∈ [-1, 1]^3
      在环境内部会乘以各自的最大速度（pursuer_max_speed / evader_max_speed）。
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 800,
        pos_limit: float = 300.0,
        catch_threshold: float = 1.0,
        pursuer_max_speed: float = 6.0,
        evader_max_speed: float = 2.0,
        initial_distance: float = 250.0,
        detection_radius: float = 200.0,
        takeoff_altitude: float = 30.0,
        takeoff_forward_speed: float | None = None,
        takeoff_climb_rate: float | None = None,
        approach_speed_scale: float = 3.0,
        max_takeoff_steps: int = 200,
    ):
        # 基本参数
        self.dt = dt
        self.max_steps = max_steps
        self.pos_limit = pos_limit
        self.catch_threshold = catch_threshold

        # 两架无人机动力学模型
        self.pursuer = DroneModel()
        self.evader = DroneModel()

        self.num_agents = 2
        self.obs_dim = 24
        self.act_dim = 3

        # 速度参数
        self.pursuer_max_speed = pursuer_max_speed
        self.evader_max_speed = evader_max_speed
        self.initial_distance = initial_distance
        self.detection_radius = detection_radius

        # 三阶段相关参数
        self.takeoff_altitude = takeoff_altitude
        self.max_takeoff_steps = max_takeoff_steps

        # 起飞阶段前向速度 & 爬升速度
        self.takeoff_forward_speed = (
            0.7 * pursuer_max_speed if takeoff_forward_speed is None else takeoff_forward_speed
        )
        self.takeoff_climb_rate = (
            0.5 * pursuer_max_speed if takeoff_climb_rate is None else takeoff_climb_rate
        )

        # 接近阶段速度倍率
        self.approach_speed = self.evader_max_speed * approach_speed_scale

        # 计数器与阶段
        self.step_count = 0
        self.caught = False
        self.phase = "approach"   # "approach" / "takeoff" / "pursuit"
        self.phase_step = 0

        # 轨迹仅用于可视化
        self.traj_p: List[np.ndarray] = []
        self.traj_e: List[np.ndarray] = []

        # 随机数种子（运行中也可以从外部重设）
        self.np_random = np.random.RandomState(0)

    # ================= 工具函数 =================

    def _get_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回追捕者和逃逸者的 12 维状态向量。"""
        s_p = self.pursuer.get_state()
        s_e = self.evader.get_state()
        return s_p, s_e

    def _build_obs(self, s_p: np.ndarray, s_e: np.ndarray) -> List[np.ndarray]:
        """
        obs_0 (pursuer): [s_p, s_e]
        obs_1 (evader): [s_e, s_p]
        """
        obs_p = np.concatenate([s_p, s_e], axis=0)
        obs_e = np.concatenate([s_e, s_p], axis=0)
        return [obs_p.astype(np.float32), obs_e.astype(np.float32)]

    def _distance(self, s_p: np.ndarray, s_e: np.ndarray) -> float:
        """三维空间距离。"""
        return float(np.linalg.norm(s_p[:3] - s_e[:3]))

    # ================= Gym 风格接口 =================

    def reset(self, seed: int | None = None) -> Tuple[List[np.ndarray], dict]:
        """重置环境，回到 approach 阶段。"""
        if seed is not None:
            self.np_random.seed(seed)

        # 追捕者：基地地面待命
        init_p = np.zeros(12, dtype=np.float32)
        init_p[0] = 0.0  # x
        init_p[1] = 0.0  # y
        init_p[2] = 0.0  # z = 地面高度

        # 逃逸者：从雷达范围外随机一个方向生成
        while True:
            dir_vec = self.np_random.normal(size=(3,))
            norm = np.linalg.norm(dir_vec) + 1e-6
            unit_vec = dir_vec / norm

            # 初始距离，保证 > detection_radius + 一定裕量
            R_min = max(self.detection_radius + 20.0, self.initial_distance)
            R_max = R_min + 50.0
            R = float(self.np_random.uniform(R_min, R_max))

            init_e = np.zeros(12, dtype=np.float32)
            init_e[:3] = init_p[:3] + unit_vec * R

            dist0 = np.linalg.norm(init_e[:3] - init_p[:3])
            if dist0 > self.detection_radius + 5.0:
                break

        self.pursuer.reset(init_p)
        self.evader.reset(init_e)

        self.step_count = 0
        self.caught = False
        self.phase = "approach"
        self.phase_step = 0

        s_p, s_e = self._get_states()
        self.traj_p = [s_p[:3].copy()]
        self.traj_e = [s_e[:3].copy()]

        obs_n = self._build_obs(s_p, s_e)
        info = {"phase": self.phase, "dist": self._distance(s_p, s_e)}
        return obs_n, info

    def step(self, actions_n: List[np.ndarray]):
        """
        单步仿真，带三阶段逻辑。
        actions_n: [action_pursuer, action_evader]，每个 shape=(3,)
        """
        self.step_count += 1
        self.phase_step += 1

        # ================= 阶段 1：approach =================
        if self.phase == "approach":
            s_p, s_e = self._get_states()

            # 追捕者静止在地面
            zero_lin = np.zeros(3, dtype=np.float32)
            zero_ang = np.zeros(3, dtype=np.float32)
            self.pursuer.set_control_input(zero_lin, zero_ang)

            # 逃逸者朝基地飞行（加速版）
            vec_to_base = s_p[:3] - s_e[:3]
            dist = np.linalg.norm(vec_to_base) + 1e-6
            dir_to_base = vec_to_base / dist
            v_e = dir_to_base * self.approach_speed
            self.evader.set_control_input(v_e.astype(np.float32), zero_ang)

            # 动力学更新
            self.pursuer.dynamics(self.dt)
            self.evader.dynamics(self.dt)

            s_p, s_e = self._get_states()
            self.traj_p.append(s_p[:3].copy())
            self.traj_e.append(s_e[:3].copy())

            dist = self._distance(s_p, s_e)
            out_of_bound = (
                np.any(np.abs(s_p[:3]) > self.pos_limit)
                or np.any(np.abs(s_e[:3]) > self.pos_limit)
            )

            # approach 阶段不用于训练，奖励为 0
            rewards_n = [0.0, 0.0]
            caught = False

            # 阶段切换：进入 takeoff
            if dist <= self.detection_radius and not out_of_bound:
                self.phase = "takeoff"
                self.phase_step = 0

            terminated = False
            truncated = out_of_bound or (self.step_count >= self.max_steps)
            done = terminated or truncated

            obs_n = self._build_obs(s_p, s_e)
            info = {
                "dist": dist,
                "caught": caught,
                "terminated": terminated,
                "truncated": truncated,
                "phase": self.phase,
            }
            return obs_n, rewards_n, done, info

        # ================= 阶段 2：takeoff =================
        if self.phase == "takeoff":
            s_p, s_e = self._get_states()

            # --- 追捕者：固定翼起飞（水平对准逃逸者 + 向上爬升） ---
            vec_pe = s_e[:3] - s_p[:3]  # from pursuer to evader
            vec_xy = vec_pe.copy()
            vec_xy[2] = 0.0
            norm_xy = np.linalg.norm(vec_xy)

            if norm_xy < 1e-6:
                dir_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                dir_xy = vec_xy / norm_xy

            v_forward = dir_xy * self.takeoff_forward_speed
            v_climb = np.array([0.0, 0.0, self.takeoff_climb_rate], dtype=np.float32)
            v_p = (v_forward + v_climb).astype(np.float32)

            zero_ang = np.zeros(3, dtype=np.float32)
            self.pursuer.set_control_input(v_p, zero_ang)

            # --- 逃逸者：从 takeoff 阶段开始可由 MADDPG 控制 ---
            a_e = np.clip(np.asarray(actions_n[1], dtype=np.float32), -1.0, 1.0)
            a_e = a_e * self.evader_max_speed
            self.evader.set_control_input(a_e, zero_ang)

            # 动力学更新
            self.pursuer.dynamics(self.dt)
            self.evader.dynamics(self.dt)

            s_p, s_e = self._get_states()
            self.traj_p.append(s_p[:3].copy())
            self.traj_e.append(s_e[:3].copy())

            dist = self._distance(s_p, s_e)
            out_of_bound = (
                np.any(np.abs(s_p[:3]) > self.pos_limit)
                or np.any(np.abs(s_e[:3]) > self.pos_limit)
            )

            # takeoff 阶段奖励仅用于监控，不写入 buffer
            scale = 0.1
            r_p = -dist * scale
            r_e = dist * scale
            rewards_n = [r_p, r_e]

            # 默认不在 takeoff 判定抓到，避免把抓捕发生在未完成起飞阶段
            caught = False

            # 起飞完成：高度达到或起飞阶段超过设定步数
            if (
                (s_p[2] >= self.takeoff_altitude or self.phase_step >= self.max_takeoff_steps)
                and (not out_of_bound)
            ):
                self.phase = "pursuit"
                self.phase_step = 0

            terminated = False
            truncated = out_of_bound or (self.step_count >= self.max_steps)
            done = terminated or truncated

            obs_n = self._build_obs(s_p, s_e)
            info = {
                "dist": dist,
                "caught": caught,
                "terminated": terminated,
                "truncated": truncated,
                "phase": self.phase,
            }
            return obs_n, rewards_n, done, info

        # ================= 阶段 3：pursuit =================
        # 此阶段由 MADDPG 控制双方，经验也只在此阶段写入 ReplayBuffer
        a_p = np.clip(np.asarray(actions_n[0], dtype=np.float32), -1.0, 1.0)
        a_e = np.clip(np.asarray(actions_n[1], dtype=np.float32), -1.0, 1.0)

        a_p = a_p * self.pursuer_max_speed
        a_e = a_e * self.evader_max_speed

        zero_ang = np.zeros(3, dtype=np.float32)
        self.pursuer.set_control_input(a_p, zero_ang)
        self.evader.set_control_input(a_e, zero_ang)

        self.pursuer.dynamics(self.dt)
        self.evader.dynamics(self.dt)

        s_p, s_e = self._get_states()
        self.traj_p.append(s_p[:3].copy())
        self.traj_e.append(s_e[:3].copy())

        dist = self._distance(s_p, s_e)
        out_of_bound = (
            np.any(np.abs(s_p[:3]) > self.pos_limit)
            or np.any(np.abs(s_e[:3]) > self.pos_limit)
        )

        scale = 0.1
        r_p = -dist * scale
        r_e = dist * scale

        caught = dist < self.catch_threshold
        if caught:
            r_p += 10.0
            r_e -= 10.0
        if out_of_bound:
            r_p -= 1.0
            r_e -= 1.0

        rewards_n = [r_p, r_e]

        terminated = caught
        truncated = out_of_bound or (self.step_count >= self.max_steps)
        done = terminated or truncated

        obs_n = self._build_obs(s_p, s_e)
        info = {
            "dist": dist,
            "caught": caught,
            "terminated": terminated,
            "truncated": truncated,
            "phase": self.phase,
        }
        return obs_n, rewards_n, done, info

    # ================= 可视化 =================

    def render(self):
        """简单三维轨迹可视化。"""
        if not self.traj_p or not self.traj_e:
            return

        traj_p = np.array(self.traj_p)
        traj_e = np.array(self.traj_e)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2], label="pursuer")
        ax.plot(traj_e[:, 0], traj_e[:, 1], traj_e[:, 2], label="evader")

        all_xyz = np.concatenate([traj_p, traj_e], axis=0)
        x_min, x_max = all_xyz[:, 0].min(), all_xyz[:, 0].max()
        y_min, y_max = all_xyz[:, 1].min(), all_xyz[:, 1].max()
        z_min, z_max = all_xyz[:, 2].min(), all_xyz[:, 2].max()

        expand = 1.2
        dx = (x_max - x_min) * expand
        dy = (y_max - y_min) * expand
        dz = (z_max - z_min) * expand

        cx = (x_max + x_min) / 2.0
        cy = (y_max + y_min) / 2.0
        cz = (z_max + z_min) / 2.0

        ax.set_xlim(cx - dx / 2.0, cx + dx / 2.0)
        ax.set_ylim(cy - dy / 2.0, cy + dy / 2.0)
        ax.set_zlim(max(0.0, cz - dz / 2.0), cz + dz / 2.0)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_wlabel("Z (m)")
        ax.legend()
        plt.show()
