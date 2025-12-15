import matplotlib.pyplot as plt


def update_plot(ax, x, y, z):
    ax.clear()
    ax.scatter(x, y, z, color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
