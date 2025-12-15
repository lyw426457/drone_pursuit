# environment/drone_env.py
import numpy as np
import gym
from gym.spaces import Box
from .drone_model import DroneModel


class DroneEnv(gym.Env):
    """
    简单的三维无人机追踪固定目标的环境（先单机，后面再扩展成追逃多机）。
    - 状态：12 维，直接用 DroneModel 的 state
    - 动作：6 维连续 [vx_cmd, vy_cmd, vz_cmd, p_cmd, q_cmd, r_cmd] ∈ [-1, 1]
    - 奖励：负的距离（越靠近目标越好）
    - 终止：
        · 距离目标 < dist_threshold
        · 飞出边界
        · 超过 max_steps
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        target_pos: np.ndarray | None = None,
        max_steps: int = 300,
        dt: float = 0.1,
        pos_limit: float = 30.0,
        dist_threshold: float = 0.5,
    ):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.pos_limit = pos_limit
        self.dist_threshold = dist_threshold

        self.drone = DroneModel()

        # 目标位置
        if target_pos is None:
            self.target_pos = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        else:
            self.target_pos = np.asarray(target_pos, dtype=np.float32)

        # 观测空间：12 维
        high_obs = np.array(
            [
                pos_limit,
                pos_limit,
                pos_limit,
                10.0,
                10.0,
                10.0,
                np.pi,
                np.pi,
                np.pi,
                10.0,
                10.0,
                10.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high_obs, high_obs, dtype=np.float32)

        # 动作空间：6 维 ∈ [-1, 1]
        self.action_space = Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_count = 0
        self.traj = []  # 存路径，便于可视化

    def _get_obs(self) -> np.ndarray:
        return self.drone.get_state()

    def _get_dist_to_target(self) -> float:
        pos = self.drone.get_state()[:3]
        return float(np.linalg.norm(pos - self.target_pos))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # 随机一点初始位置，靠近原点
        init_pos = self.np_random.uniform(low=-2.0, high=2.0, size=(3,))
        init_state = np.zeros(12, dtype=np.float32)
        init_state[:3] = init_pos
        self.drone.reset(init_state)

        self.step_count = 0
        self.traj = [self.drone.get_state()[:3].copy()]

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        # clip 到动作空间
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 前三维线速度，后三维角速度
        self.drone.set_control_input(action[:3], action[3:])
        self.drone.dynamics(self.dt)

        obs = self._get_obs()
        pos = obs[:3]

        # 路径记录
        self.traj.append(pos.copy())

        # 距离目标的负值作为奖励
        dist = self._get_dist_to_target()
        reward = -dist

        # 约束：飞太远 / 超高度也惩罚
        out_of_bound = np.any(np.abs(pos) > self.pos_limit)
        if out_of_bound:
            reward -= 10.0

        # 终止条件
        terminated = dist < self.dist_threshold
        truncated = out_of_bound or (self.step_count >= self.max_steps)

        done = terminated or truncated
        info = {"dist": dist, "terminated": terminated, "truncated": truncated}

        return obs, reward, done, info

    def render(self, mode="human"):
        """
        用 matplotlib 画出当前轨迹（一次性静态图）。
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 触发 3D 投影注册

        if not self.traj:
            return

        traj = np.stack(self.traj, axis=0)
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, "-b", label="drone path")
        ax.scatter(
            [self.target_pos[0]],
            [self.target_pos[1]],
            [self.target_pos[2]],
            c="r",
            marker="*",
            s=80,
            label="target",
        )
        ax.set_xlim(-self.pos_limit, self.pos_limit)
        ax.set_ylim(-self.pos_limit, self.pos_limit)
        ax.set_zlim(-self.pos_limit, self.pos_limit)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
