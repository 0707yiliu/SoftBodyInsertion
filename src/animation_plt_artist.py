import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = plt.plot([], [], "r-", animated=True)
x = []
y = []
nums = 100
def init():
    ax.set_xlim(1, nums)
    ax.set_ylim(-1, 1)
    return line,

def update(frame):
    # print(frame)
    x.append(frame)
    y.append(np.sin(frame))
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig
                   ,update
                   ,frames=np.linspace(0, nums-1 ,nums)
                   ,interval=100
                   ,init_func=init
                   ,blit=True
                   )
ani.save("exp_video" + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])