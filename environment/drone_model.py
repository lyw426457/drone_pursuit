# environment/drone_model.py

import numpy as np


class DroneModel:
    """
    一个极简的四旋翼动力学近似：
    - 状态 state: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    - 控制 control_input: [vx_cmd, vy_cmd, vz_cmd, p_cmd, q_cmd, r_cmd]
      这里直接把控制指令当作速度/角速度的“期望值”，用一阶响应去跟踪。
    """

    def __init__(
        self,
        mass: float = 1.0,
        Ixx: float = 0.1,
        Iyy: float = 0.1,
        Izz: float = 0.1,
        vel_tau: float = 0.3,
        ang_tau: float = 0.2,
    ):
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

        # 状态：[x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.state = np.zeros(12, dtype=np.float32)

        # 控制输入：[vx_cmd, vy_cmd, vz_cmd, p_cmd, q_cmd, r_cmd]
        self.control_input = np.zeros(6, dtype=np.float32)

        # 一阶响应时间常数（越小越“灵”）
        self.vel_tau = vel_tau
        self.ang_tau = ang_tau

    def reset(self, init_state: np.ndarray | None = None):
        if init_state is None:
            self.state = np.zeros(12, dtype=np.float32)
        else:
            assert init_state.shape == (12,)
            self.state = init_state.astype(np.float32)

    def set_control_input(self, linear_velocity: np.ndarray, angular_velocity: np.ndarray):
        """
        linear_velocity: shape (3,) -> [vx_cmd, vy_cmd, vz_cmd]
        angular_velocity: shape (3,) -> [p_cmd, q_cmd, r_cmd]
        """
        linear_velocity = np.asarray(linear_velocity, dtype=np.float32)
        angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
        assert linear_velocity.shape == (3,)
        assert angular_velocity.shape == (3,)

        # 这里是“拼接”，不是逐元素加
        self.control_input = np.concatenate([linear_velocity, angular_velocity], axis=0)

    def dynamics(self, dt: float):
        """
        用简单的一阶系统逼近期望速度/角速度，然后积分得到位姿。
        真正精细的四旋翼动力学以后可以再换成更复杂的。
        """
        x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r = self.state
        vx_cmd, vy_cmd, vz_cmd, p_cmd, q_cmd, r_cmd = self.control_input

        # 一阶逼近：dv/dt = (v_cmd - v) / tau
        vx += (vx_cmd - vx) * dt / self.vel_tau
        vy += (vy_cmd - vy) * dt / self.vel_tau
        vz += (vz_cmd - vz) * dt / self.vel_tau

        p += (p_cmd - p) * dt / self.ang_tau
        q += (q_cmd - q) * dt / self.ang_tau
        r += (r_cmd - r) * dt / self.ang_tau

        # 位置积分
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # 姿态积分（欧拉角）
        roll += p * dt
        pitch += q * dt
        yaw += r * dt

        # 限制高度不要太离谱（防止数值爆炸）
        z = np.clip(z, -20.0, 20.0)

        self.state = np.array(
            [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r], dtype=np.float32
        )

    def get_state(self) -> np.ndarray:
        return self.state.copy()
