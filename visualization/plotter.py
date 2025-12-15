# visualization/plotter.py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_trajectory(traj: np.ndarray, target: np.ndarray | None = None):
    """
    traj: (T, 3)
    target: (3,)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
    ax.plot(xs, ys, zs, "-b", label="drone path")

    if target is not None:
        ax.scatter([target[0]], [target[1]], [target[2]], c="r", marker="*", s=80, label="target")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
