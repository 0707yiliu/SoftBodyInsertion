import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from scipy import signal
import time
wn = 5*5/500 # cutting-frequency
b, a = signal.butter(4,wn,'low')
# plt animation function (2-D)
row = 6
col = 2  # how much row and column the fig need,
# for this project, it needs ee-pos, ee-rot, force and torque (6x2)
# fig-size
w = 9
h = 10.5
ylabels_vision_touch = ["EEF-x", "EEF-y", "EEF-z", "EEF-rx", "EEF-ry", "EEF-rz",
                        "F-x", "F-y", "F-z", "tau-x", "tau-y", "tau-z"]
# color_set = ["deepskyblue", "royalblue", "gold", "springgreen", "green", "darkgrey", "darksalmon", "red", "black", "chocolate", "magenta", "darkorange"]
color_set = ["green", "red", "blue"] # for assembly
fig, ax = plt.subplots(row, col, figsize=(w, h))
# data_list = [
#              "visiontouch0805170118_real_dsl",
#              "visiontouch0805173028_real_dsl",
#              "visiontouch0805173211_real_dsl",
#              "visiontouch0805173329_real_dsl",
#              "visiontouch0805173427_real_dsl",
#              "visiontouch0807135502_real_nodsl",
#              #
#              "visiontouch0805170426_real_dsl",
#              "visiontouch0805173752_real_dsl",
#              "visiontouch0805173854_real_dsl",
#              "visiontouch0805173944_real_dsl",
#              "visiontouch0727162142_real_dsl",
#              "visiontouch0727162254_real_dsl"
#              ] # for exait

# data_list = [
#             "visiontouch0727170639_real_dsl",
#             "visiontouch0727171016_real_dsl",
#             "visiontouch0805160920_real_dsl",
#             "visiontouch0805165327_real_dsl",
#             "visiontouch0805173028_real_dsl",
#             "visiontouch0805174415_real_dsl",
#
#             "visiontouch0727170801_real_dsl",
#             "visiontouch0727170906_real_dsl",
#             "visiontouch0805150608_real_dsl",
#             "visiontouch0805150952_real_dsl",
#             "visiontouch0805172844_real_dsl",
#             "visiontouch0807140126_real_nodsl"
#
# ] # for softbody DR

data_list = [
    "visiontouch0727174741_real_nodsl",
    "visiontouch0727162030_real_dsl",
    "visiontouch0727162254_real_dsl"
]

# data_list = ["visiontouch0727170639_real_dsl",
#              "visiontouch0727171016_real_dsl",
#              "visiontouch0805160920_real_dsl",
#              "visiontouch0805165327_real_dsl",
#              "visiontouch0805173028_real_dsl",
#              "visiontouch0805174415_real_dsl",
#              "visiontouch0727170801_real_dsl",
#              "visiontouch0727170906_real_dsl",
#              "visiontouch0805150608_real_dsl",
#              "visiontouch0805150952_real_dsl",
#              "visiontouch0805172844_real_dsl",
#              "visiontouch0807140126_real_nodsl"
#              ] # for DR
exp_video = "hole1"
tool_num = len(data_list)
root = "/home/yi/project_ghent/recording/"
data_file = list()

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
max_len = 0
col_item = 0
# for i in range(num_line * num_set):
#     line.append(Line2D([], [], color=color_set[col_item-1], linewidth=1.75))
#     if i % num_line == 0:
#         col_item += 1
#         print(col_item)

for j in range(num_set):
    for k in range(num_line):
        line.append(Line2D([], []))
        if k <= 5:
            x = k
            y = 0
        else:
            x = k - 6
            y = 1
        item = k + j * num_line
        line[item], = ax[x, y].plot([], [], color=color_set[j], linewidth=1.75)
        ax[x, y].set_ylabel(ylabels_vision_touch[k])
        ax[x, y].spines['top'].set_visible(False)
        ax[x, y].spines['right'].set_visible(False)
        if k == 5 or k == 11:
            ax[x, y].set_xlabel("Step")
        if k == 2 or k == 3 or k == 4 or k == 5 or k == 6 or k == 7:
            ax[x, y].set_xlabel("Step")
for j in range(num_set):
    # datasets
    data_dir = root + data_list[j] + "/" + 'realworld.npy'
    obs = np.load(data_dir)
    data_len = len(obs)
    print(data_len)
    if data_len > max_len:
        max_len = data_len
# for j in range(num_set):
#     # datasets
#     data_dir = root + data_list[j] + "/" + 'realworld.npy'
#     obs = np.load(data_dir)
#     data_len = len(obs)
#     print(data_len)
#     if data_len > max_len:
#         max_len = data_len
#     for i in range(obs.shape[1]):
#         obs[:, i] = signal.filtfilt(b, a, obs[:, i])
#     for k in range(num_line):
#         # 12 types of data in each dataset
#         if k <= 5:
#             x = k
#             y = 0
#         else:
#             x = k - 6
#             y = 1
#         # ax[x][y].clear()
#         item = k + j * num_line
#
#         # line[item], = ax[x,y].plot(np.linspace(0, max_len - 1, max_len), obs_buffer[j, :, k], color=color_set[j])
#         line[item], = ax[x, y].plot(np.linspace(0, data_len - 1, data_len), obs[:, k])
#         ax[x, y].set_ylabel(ylabels_vision_touch[k])
#         ax[x, y].spines['top'].set_visible(False)
#         ax[x, y].spines['right'].set_visible(False)
#         # print((j+1)*(k+1)-1)
#         # ydata[j, i, k] = np.copy(obs[j, i, k])
#         if k == 5 or k == 11:
#             ax[x, y].set_xlabel("Step")
#         if k == 2 or k == 3 or k == 4 or k == 5 or k == 6 or k == 7:
#             ax[x, y].set_xlabel("Step")
#             ax[x, y].set_ylim([-1.2, 1.2])
#         else:
#             ax[x, y].set_ylim([-1.2, 1.2])


def init_ani():
    for j in range(num_set):
        # 4 datasets
        for k in range(num_line):
            if k <= 5:
                x = k
                y = 0
            else:
                x = k - 6
                y = 1
            ax[x, y].set_ylim([-1.2, 1.2])
            ax[x, y].set_xlim([-100, max_len+100])
    return line
for j in range(num_set):
    # datasets
    data_dir = root + data_list[j] + "/" + 'realworld.npy'
    obs = np.load(data_dir)
    for m in range(obs.shape[1]):
        obs[:, m] = signal.filtfilt(b, a, obs[:, m])
    data_len = len(obs)
    if j == 0:
        obs1 = obs
    elif j == 1:
        obs2 = obs
    elif j == 2:
        obs3 = obs
    elif j == 3:
        obs4 = obs
    elif j == 4:
        obs5 = obs
    elif j == 5:
        obs6 = obs
    elif j == 6:
        obs7 = obs
    elif j == 7:
        obs8 = obs
    elif j == 8:
        obs9 = obs
    elif j == 9:
        obs10 = obs
    elif j == 10:
        obs11 = obs
    elif j == 11:
        obs12 = obs

def animate(i):
    # ax[:,:].cla()
    # plt.cla()
    print(i)
    i = int(i)
    for j in range(num_set):
        # datasets
        if j == 0:
            obs = obs1
        elif j == 1:
            obs = obs2
        elif j == 2:
            obs = obs3
        elif j == 3:
            obs = obs4
        elif j == 4:
            obs = obs5
        elif j == 5:
            obs = obs6
        elif j == 6:
            obs = obs7
        elif j == 7:
            obs = obs8
        elif j == 8:
            obs = obs9
        elif j == 9:
            obs = obs10
        elif j == 10:
            obs = obs11
        elif j == 11:
            obs = obs12
        for k in range(num_line):
            if k <= 5:
                x = k
                y = 0
            else:
                x = k - 6
                y = 1
            item = k + j * num_line
            # ax[x, y].cla()
            if i > len(obs):
            # line[item], = ax[x,y].plot(np.linspace(0, max_len - 1, max_len), obs_buffer[j, :, k], color=color_set[j])
            #     line[item], = ax[x, y].plot(np.linspace(0, data_len - 1, data_len), obs[:, k], color=color_set[j], linewidth=1.75)
                pass
            else:
                # x = list[:i]
                # print(x)
                # line[item], = ax[x, y].plot(np.linspace(0, i-1, i), obs[:i, k])
                m = list(np.linspace(0, i-1, i))
                n = list(obs[:i, k])
                line[item].set_data(m, n)
            # print(line[0])
    return line

#
ani = animation.FuncAnimation(
                            fig=fig,
                            func=animate,
                            frames=np.linspace(0 ,max_len-1, max_len),
                            init_func=init_ani,
                            interval=100,
                            blit=True)
ani.save(exp_video + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
