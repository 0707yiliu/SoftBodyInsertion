import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from scipy import signal
import time
wn = 8*5/500 # cutting-frequency
b, a = signal.butter(4,wn,'low')
# plt animation function (2-D)
row = 6
col = 2  # how much row and column the fig need,
# for this project, it needs ee-pos, ee-rot, force and torque (6x2)
# fig-size
w = 9
h = 20
ylabels_vision_touch = ["EEF-x", "EEF-y", "EEF-z", "EEF-rx", "EEF-ry", "EEF-rz",
                        "F-x", "F-y", "F-z", "tau-x", "tau-y", "tau-z"]
color_set = ["deepskyblue", "royalblue", "gold", "springgreen", "green", "darkgrey", "darksalmon", "red", "black", "chocolate", "magenta", "darkorange"]
fig, ax = plt.subplots(row, col, figsize=(w, h))
# data_list = ["visiontouch0805170426_real_dsl",
#              "visiontouch0805173028_real_dsl",
#              "visiontouch0805173211_real_dsl",
#              "visiontouch0805173329_real_dsl",
#              "visiontouch0805173427_real_dsl",
#              "visiontouch0805173752_real_dsl",
#              "visiontouch0805173854_real_dsl",
#              "visiontouch0805173944_real_dsl",
#              "visiontouch0805174415_real_dsl",
#              "visiontouch0805170118_real_dsl",
#              "visiontouch0807135502_real_nodsl",
#              "visiontouch0807135852_real_nodsl"
#              ] # for exait
data_list = ["visiontouch0727170639_real_dsl",
             "visiontouch0727171016_real_dsl",
             "visiontouch0805160920_real_dsl",
             "visiontouch0805165327_real_dsl",
             "visiontouch0805173028_real_dsl",
             "visiontouch0805174415_real_dsl",
             "visiontouch0727170801_real_dsl",
             "visiontouch0727170906_real_dsl",
             "visiontouch0805150608_real_dsl",
             "visiontouch0805150952_real_dsl",
             "visiontouch0805172844_real_dsl",
             "visiontouch0807140126_real_nodsl"
             ] # for DR
exp_video = "dsl_hole1Level2"
tool_num = len(data_list)
root = "/home/yi/project_ghent/recording/"
data_file = list()
max_len = 0

# # get maximum length and load data into obs_buffer
# for i in range(tool_num):
#     data_dir = root + data_list[i] + "/" + 'realworld.npy'
#     data_file.append(data_dir)
#     obs = np.load(data_file[i])
#     if obs.shape[0] > max_len:
#         max_len = obs.shape[0]
# obs_buffer = np.zeros((tool_num, max_len, row * col))
# for i in range(tool_num):
#     obs = np.load(data_file[i])
#     obs_pad = np.pad(obs, ((0, max_len - obs.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
#     for j in range(obs_pad.shape[1]):
#         obs_pad[:, j] = signal.filtfilt(b, a, obs_pad[:, j])
#     obs_buffer[i, :, :] = obs_pad[:, :row * col]

# create line
num_line = row * col
num_set = tool_num
line = []
for i in range(num_line * num_set):
    line.append(Line2D([], []))
for j in range(num_set):
    # datasets
    data_dir = root + data_list[j] + "/" + 'realworld.npy'
    obs = np.load(data_dir)
    data_len = len(obs)
    for i in range(obs.shape[1]):
        obs[:, i] = signal.filtfilt(b, a, obs[:, i])
    for k in range(num_line):
        # 12 types of data in each dataset
        if k <= 5:
            x = k
            y = 0
        else:
            x = k - 6
            y = 1
        # ax[x][y].clear()
        item = k + j * num_line

        # line[item], = ax[x,y].plot(np.linspace(0, max_len - 1, max_len), obs_buffer[j, :, k], color=color_set[j])
        line[item], = ax[x, y].plot(np.linspace(0, data_len - 1, data_len), obs[:, k], color=color_set[j], linewidth=1.75)
        ax[x, y].set_ylabel(ylabels_vision_touch[k])
        ax[x, y].spines['top'].set_visible(False)
        ax[x, y].spines['right'].set_visible(False)
        # print((j+1)*(k+1)-1)
        # ydata[j, i, k] = np.copy(obs[j, i, k])
        if k == 5 or k == 11:
            ax[x, y].set_xlabel("Step")
        if k == 2 or k == 3 or k == 4 or k == 5 or k == 6 or k == 7:
            ax[x, y].set_xlabel("Step")
            ax[x, y].set_ylim([-1.2, 1.2])
        else:
            ax[x, y].set_ylim([-1.2, 1.2])


def init_ani():
    for j in range(num_set):
        # 4 datasets
        for k in range(num_line):
            item = k + j * num_line
            line[item].set_data([], [])


def animate(i):
    # # ax.clear()
    # plt.cla()
    for j in range(num_set):
        # 4 datasets
        for k in range(num_line):
            # 12 types of data in each dataset

            # ax[k].plot(np.linspace(0, obs_max_len - 1, obs_max_len), obs[j, :, k], color=color_set[j], label='eef-x')
            # if k <= 5:
            #     x = k
            #     y = 0
            # else:
            #     x = k - 6
            #     y = 1
            item = k + j * num_line
            line[item].set_data(np.linspace(0, i, i), obs[j, :i, k])
    #         # ax[x][y].clear()
    #         ln, = ax[x,y].plot(np.linspace(0, i, i), obs[j, :i, k], linewidth=1, color=color_set[j], label='eef-x')
    #         # ydata[j, i, k] = np.copy(obs[j, i, k])
    #         # # print(type(ydata[j, :i, k].tolist()))
    #         # ln.set_ydata(ydata[j, :, k])
    #         # print(np.linspace(0, obs_max_len - 1, obs_max_len)[i])
    #         if k == 2 or 3 or 4 or 5 or 6 or 7:
    #             ax[x,y].set_ylim([-2.3, 2.3])
    #         else:
    #             ax[x,y].set_ylim([-1.2, 1.2])
    return line


# ani = animation.FuncAnimation(fig=fig, func=animate, frames=max_len, init_func=init_ani,
#                               interval=100, repeat=True)
# ani.save(exp_video + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
