# drone_pursuit/visualization/acmi_export.py

import os
import math
import datetime
from typing import Sequence

import numpy as np


def _xy_to_latlon(x: float, y: float, lat0: float, lon0: float):
    """
    简单平面近似：将以 (0,0) 为原点的平面坐标 (x,y) [单位: m]
    转成经纬度 (lat, lon)，以 (lat0, lon0) 为参考点。
    """
    # 1 度纬度约 111 km
    dlat = y / 111000.0
    # 1 度经度约 cos(lat)*111 km
    dlon = x / (111000.0 * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def export_traj_to_acmi(
    traj_p: Sequence[Sequence[float]],
    traj_e: Sequence[Sequence[float]],
    dt: float,
    filepath: str,
    lat0: float = 1.3000,
    lon0: float = 103.8000,
    name_pursuer: str = "Pursuer",
    name_evader: str = "Evader",
    base_altitude: float = 1000.0,   # 新增：统一抬高到海拔 1000m 上
    xy_scale: float = 1.0,           # 新增：只在可视化时放大水平距离
):
    """
    将一条 episode 的轨迹导出为 Tacview 2.2 ACMI 文本文件。

    参数
    ----
    traj_p : 追捕者轨迹，形如 [(x,y,z), ...] 的序列
    traj_e : 逃逸者轨迹，形如 [(x,y,z), ...] 的序列
    dt     : 仿真步长（秒）
    filepath : 输出 .acmi 文件完整路径
    lat0, lon0 : 作为 (x=0,y=0) 的参考经纬度
    name_pursuer : Tacview 中显示的追捕者名称
    name_evader  : Tacview 中显示的逃逸者名称
    base_altitude : 所有目标统一加的基准高度（防止飞到地底）
    xy_scale      : 只对 x、y 做缩放的系数，用于放大水平运动尺度
    """
    if not traj_p or not traj_e:
        print("[ACMI] 轨迹为空，无法导出。")
        return

    # 保证长度一致
    n = min(len(traj_p), len(traj_e))
    traj_p = np.array(traj_p[:n], dtype=float)
    traj_e = np.array(traj_e[:n], dtype=float)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        # ===== 1. 文件头（必须） =====
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")

        # 全局对象 id=0，只用来写 ReferenceTime 等全局属性
        ref_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        f.write(f"0,ReferenceTime={ref_time}\n\n")

        # ===== 2. 定义两个飞行器对象：用 id=1 和 id=2（注意不能用0） =====
        # 先用第 0 帧的位置初始化 T、Name、Type
        px0, py0, pz0 = traj_p[0]
        ex0, ey0, ez0 = traj_e[0]

        # 只在可视化层面放大 x,y，不改变 z
        px0_vis = px0 * xy_scale
        py0_vis = py0 * xy_scale
        ex0_vis = ex0 * xy_scale
        ey0_vis = ey0 * xy_scale

        plat0, plon0 = _xy_to_latlon(px0_vis, py0_vis, lat0, lon0)
        elat0, elon0 = _xy_to_latlon(ex0_vis, ey0_vis, lat0, lon0)

        # altitude = 基准高度 + max(z, 0)
        palt0 = base_altitude + max(pz0, 0.0)
        ealt0 = base_altitude + max(ez0, 0.0)

        # 时间帧 #0.0
        f.write("#0.0\n")

        # ====== 追捕者：id=1 ======
        f.write(
            "1,"
            f"T={plon0:.7f}|{plat0:.7f}|{palt0:.1f},"
            "Name=F-16C,"             # 用真实机型加载3D模型
            "Callsign=Pursuer,"       
            "Type=Air+FixedWing,"
            "Country=us,"
            "Coalition=Allies,"
            "Color=Blue\n"
        )

        # ====== 逃逸者：id=2 ======
        f.write(
            "2,"
            f"T={elon0:.7f}|{elat0:.7f}|{ealt0:.1f},"
            "Name=MiG-29,"
            "Callsign=Evader,"
            "Type=Air+FixedWing,"
            "Country=ru,"
            "Coalition=Enemies,"
            "Color=Red\n"
        )

        # ===== 3. 后续每一帧，只更新 T 属性 =====
        for idx in range(1, n):
            t = idx * dt
            px, py, pz = traj_p[idx]
            ex, ey, ez = traj_e[idx]

            px_vis = px * xy_scale
            py_vis = py * xy_scale
            ex_vis = ex * xy_scale
            ey_vis = ey * xy_scale

            plat, plon = _xy_to_latlon(px_vis, py_vis, lat0, lon0)
            elat, elon = _xy_to_latlon(ex_vis, ey_vis, lat0, lon0)

            palt = base_altitude + max(pz, 0.0)
            ealt = base_altitude + max(ez, 0.0)

            f.write(f"\n#{t:.3f}\n")
            f.write(f"1,T={plon:.7f}|{plat:.7f}|{palt:.1f}\n")
            f.write(f"2,T={elon:.7f}|{elat:.7f}|{ealt:.1f}\n")

    print(f"[ACMI] 导出完成: {filepath}")
